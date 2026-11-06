"""Adapters expose class-1 probabilities from the two framework backbones."""
import numpy as np

from data_pipeline import augment, load_image


def batches(samples, size, rng, shuffle=False, augmentation=False):
    indices = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
    for start in range(0, len(indices), size):
        selected = [samples[i] for i in indices[start:start + size]]
        images = [load_image(s.path) for s in selected]
        if augmentation:
            images = [augment(image, rng) for image in images]
        yield np.stack(images), np.array([s.label for s in selected], dtype=np.float32)


class VGG:
    def __init__(self, weights=False, checkpoint=None):
        import tensorflow as tf

        self.tf = tf
        if checkpoint:
            self.model = tf.keras.models.load_model(checkpoint, compile=False)
        else:
            backbone = tf.keras.applications.VGG16(
                weights="imagenet" if weights else None, include_top=False, input_shape=(224, 224, 3)
            )
            backbone.trainable = not weights
            self.model = tf.keras.Sequential([
                backbone, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(1)
            ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    def inputs(self, images):
        return self.tf.keras.applications.vgg16.preprocess_input(
            np.repeat(images[..., None], 3, axis=-1).astype(np.float32)
        )

    def train(self, images, labels):
        self.model.train_on_batch(self.inputs(images), labels[:, None])

    def predict(self, images):
        return self.tf.sigmoid(self.model(self.inputs(images), training=False)).numpy().ravel()

    def save(self, path):
        self.model.save(path)


class Alex:
    def __init__(self, weights=False, checkpoint=None):
        import torch
        from torchvision.models import AlexNet_Weights, alexnet

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT if weights else None)
        if weights:
            self.model.requires_grad_(False)
        self.model.classifier[6] = torch.nn.Linear(4096, 1)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=1e-4)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def inputs(self, images):
        tensor = self.torch.as_tensor(images, device=self.device).float()[:, None] / 255
        return (tensor - self.mean) / self.std

    def train(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.inputs(images)).flatten()
        self.loss(logits, self.torch.as_tensor(labels, device=self.device)).backward()
        self.optimizer.step()

    def predict(self, images):
        self.model.eval()
        with self.torch.inference_mode():
            return self.model(self.inputs(images)).sigmoid().flatten().cpu().numpy()

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)
