"""Reproducible held-out evaluation for the synthetic character dataset."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def render_char(char: str, seed: int) -> np.ndarray:
    canvas = np.zeros((40, 40), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (tw, th), _ = cv2.getTextSize(char, font, scale, thickness)
    x = (40 - tw) // 2
    y = (40 + th) // 2
    cv2.putText(canvas, char, (x, y), font, scale, 255, thickness)

    image = cv2.resize(canvas, (20, 20))
    rng = np.random.default_rng(seed)
    noise = rng.integers(-25, 25, size=image.shape).astype(np.int8)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def build_dataset(per_char: int) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for class_index, char in enumerate(CHARS):
        for sample_index in range(per_char):
            images.append(render_char(char, class_index * 1000 + sample_index))
            labels.append(char)

    X = np.array([image.flatten().astype(np.float32) for image in images])
    return X, np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-char", type=int, default=50)
    args = parser.parse_args()

    X, y = build_dataset(args.per_char)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = SVC(kernel="rbf", gamma="scale", C=1.0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Classes: {len(CHARS)}")
    print(f"Total samples: {len(y)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print()
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
