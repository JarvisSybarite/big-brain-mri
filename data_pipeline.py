"""Read 2D exports and split by patient before any augmentation."""
import csv
from hashlib import file_digest
from os.path import relpath
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

SIZE = 224
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Sample:
    path: Path
    patient: str
    label: int
    split: str


def read_manifest(path):
    path = Path(path).resolve()
    samples, patients, files = [], {}, set()
    image_splits = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image = (path.parent / row["path"]).resolve(strict=True)
            patient, label, split = row["patient_id"].strip(), int(row["label"]), row["split"].strip()
            if not patient or label not in (0, 1) or split not in SPLITS:
                raise ValueError(f"Invalid manifest row for {image.name}")
            if image in files or patients.setdefault(patient, (label, split)) != (label, split):
                raise ValueError("Repeated file or inconsistent patient label/split")
            with image.open("rb") as image_file:
                digest = file_digest(image_file, "sha256").digest()
            if image_splits.setdefault(digest, split) != split:
                raise ValueError("Identical image content appears in different splits")
            files.add(image)
            samples.append(Sample(image, patient, label, split))
    for split in SPLITS:
        if {s.label for s in samples if s.split == split} != {0, 1}:
            raise ValueError(f"{split} must contain both labels")
    return samples


def split_manifest(source, destination, seed):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    patients, files = {}, set()
    for row in rows:
        patient = row["patient_id"].strip()
        label = int(row["label"])
        path = (source.parent / row["path"]).resolve(strict=True)
        if not patient or label not in (0, 1) or patients.setdefault(patient, label) != label or path in files:
            raise ValueError("Invalid patient, label, or repeated image")
        files.add(path)
        row.update(path=relpath(path, destination.parent), patient_id=patient, label=label)
    ids = sorted(patients)
    train, remainder = train_test_split(
        ids, test_size=0.30, random_state=seed, stratify=[patients[p] for p in ids]
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=seed, stratify=[patients[p] for p in remainder]
    )
    if any({patients[p] for p in group} != {0, 1} for group in (train, validation, test)):
        raise ValueError("Need more independent patients from both classes")
    assignments = {patient: split for split, group in zip(SPLITS, (train, validation, test)) for patient in group}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "patient_id", "label", "split"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(row, split=assignments[row["patient_id"]]) for row in rows)


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    h, w = image.shape
    scale = SIZE / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    dh, dw = SIZE - resized.shape[0], SIZE - resized.shape[1]
    return cv2.copyMakeBorder(resized, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_CONSTANT)


def augment(image, rng):
    matrix = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rng.uniform(-8, 8), rng.uniform(.97, 1.03))
    rotated = cv2.warpAffine(image, matrix, (SIZE, SIZE))
    return np.clip(rotated.astype(np.float32) * rng.uniform(.9, 1.1), 0, 255).astype(np.uint8)
