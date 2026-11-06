"""Command line interface for training, validation, testing and inference."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from data_pipeline import load_image, read_manifest, split_manifest
from models import Alex, VGG, batches

MODELS = {"vgg": (VGG, "vgg.keras"), "alex": (Alex, "alex.pt")}


def metrics(labels, scores, threshold):
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel())
    return dict(threshold=threshold, accuracy=(tp + tn) / len(labels), sensitivity=tp / (tp + fn),
                specificity=tn / (tn + fp), false_positive_rate=fp / (fp + tn),
                roc_auc=roc_auc_score(labels, scores), average_precision=average_precision_score(labels, scores),
                confusion_matrix=dict(tn=tn, fp=fp, fn=fn, tp=tp))


def cohort(samples, split):
    return [sample for sample in samples if sample.split == split]


def scores(model, samples, batch):
    return np.concatenate([model.predict(images) for images, _ in batches(samples, batch, None)])


def patient_scores(samples, probabilities):
    patients = {}
    for sample, score in zip(samples, probabilities):
        patients.setdefault(sample.patient, (sample.label, []))[1].append(score)
    return (
        np.array([label for label, _ in patients.values()]),
        np.array([np.mean(values) for _, values in patients.values()]),
    )


def train(args):
    samples = read_manifest(args.manifest)
    training, validation = cohort(samples, "train"), cohort(samples, "validation")
    if args.model == "vgg":
        import tensorflow as tf
        tf.keras.utils.set_random_seed(args.seed)
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        import torch
        torch.manual_seed(args.seed)
    factory, filename = MODELS[args.model]
    model = factory(weights=args.imagenet)
    args.run.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run / filename
    if checkpoint.exists():
        raise ValueError(f"Checkpoint already exists: {checkpoint}")
    rng = np.random.default_rng(args.seed)
    best = -1
    for epoch in range(1, args.epochs + 1):
        for images, labels in batches(training, args.batch, rng, True, args.augment):
            model.train(images, labels)
        labels, probabilities = patient_scores(validation, scores(model, validation, args.batch))
        auc = roc_auc_score(labels, probabilities)
        print(json.dumps({"epoch": epoch, "validation_patient_roc_auc": auc}), flush=True)
        if auc > best:
            model.save(checkpoint)
            best = auc


def load_models(run):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    return {name: factory(checkpoint=run / filename) for name, (factory, filename) in MODELS.items()}


def threshold_for(labels, probabilities, minimum_sensitivity):
    positives = np.sort(probabilities[labels == 1])[::-1]
    return float(positives[int(np.ceil(len(positives) * minimum_sensitivity)) - 1])


def evaluate(args):
    samples = read_manifest(args.manifest)
    models = load_models(args.run)
    validation, test = cohort(samples, "validation"), cohort(samples, "test")
    validation_predictions = {name: scores(model, validation, args.batch) for name, model in models.items()}
    validation_predictions["ensemble"] = np.mean(list(validation_predictions.values()), axis=0)
    thresholds = {}
    for name, probability in validation_predictions.items():
        labels, probability = patient_scores(validation, probability)
        thresholds[name] = threshold_for(labels, probability, args.sensitivity)
    test_predictions = {name: scores(model, test, args.batch) for name, model in models.items()}
    test_predictions["ensemble"] = np.mean(list(test_predictions.values()), axis=0)
    report = {"unit": "patient_mean_of_slice_scores", "validation_sensitivity_target": args.sensitivity, "test": {}}
    for name, probability in test_predictions.items():
        labels, probability = patient_scores(test, probability)
        report["test"][name] = metrics(labels, probability, thresholds[name])
    report["test_patients"] = int(len(labels))
    (args.run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def predict(args):
    models = load_models(args.run)
    image = load_image(args.image)[None]
    predictions = {name: float(model.predict(image)[0]) for name, model in models.items()}
    print(json.dumps({"unit": "single_slice", "scores": predictions,
                      "ensemble_score": sum(predictions.values()) / len(predictions)}, indent=2))


def benchmark(args):
    models = load_models(args.run)
    image = load_image(args.image)[None]

    def infer():
        return [model.predict(image) for model in models.values()]

    for _ in range(3):
        infer()
    times = []
    for _ in range(args.repeats):
        start = perf_counter()
        infer()
        times.append((perf_counter() - start) * 1000)
    print(json.dumps({"scope": "batch-1 ensemble forward pass and framework conversion, excluding image decoding",
                      "runs": len(times), "median_ms": float(np.median(times)),
                      "p95_ms": float(np.percentile(times, 95))}, indent=2))


def synthetic(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    rows = ["path,patient_id,label"]
    for label in (0, 1):
        for index in range(12):
            patient = f"synthetic-{label}-{index:02d}"
            filename = f"{patient}.png"
            image = np.zeros((128, 128), dtype=np.uint8)
            cv2.ellipse(image, (64, 64), (43, 50), 0, 0, 360, 90, -1)
            if label:
                cv2.circle(image, tuple(rng.integers(45, 84, size=2).tolist()), 8, 210, -1)
            image = np.clip(image.astype(float) + rng.normal(0, 9, image.shape), 0, 255).astype(np.uint8)
            if not cv2.imwrite(str(directory / filename), image):
                raise OSError(f"Failed to write {filename}")
            rows.append(f"{filename},{patient},{label}")
    (directory / "images.csv").write_text("\n".join(rows) + "\n")
    print(f"Synthetic smoke-test data only: {directory / 'images.csv'}")


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("synthetic")
    demo.add_argument("output", type=Path)
    demo.add_argument("--seed", type=int, default=42)
    split = commands.add_parser("split")
    split.add_argument("input", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--seed", type=int, default=42)
    for command in ("train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("manifest", type=Path)
        sub.add_argument("run", type=Path)
        sub.add_argument("--batch", type=positive, default=8)
        if command == "train":
            sub.add_argument("--model", choices=MODELS, required=True)
            sub.add_argument("--epochs", type=positive, default=5)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--imagenet", action="store_true")
            sub.add_argument("--augment", action="store_true")
        else:
            sub.add_argument("--sensitivity", type=float, default=.90)
    for command in ("predict", "benchmark"):
        sub = commands.add_parser(command)
        sub.add_argument("run", type=Path)
        sub.add_argument("image", type=Path)
        if command == "benchmark":
            sub.add_argument("--repeats", type=positive, default=30)
    args = parser.parse_args()
    if args.command == "split":
        split_manifest(args.input, args.output, args.seed)
        return
    if args.command == "evaluate" and not 0 < args.sensitivity <= 1:
        parser.error("--sensitivity must be in (0, 1]")
    actions = {"synthetic": synthetic, "train": train, "evaluate": evaluate, "predict": predict, "benchmark": benchmark}
    actions[args.command](args)


if __name__ == "__main__":
    main()
