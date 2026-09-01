"""Synthesize labeled character images for training the recognizer.

Renders each alphanumeric character as a 20x20 grayscale image using the
system font, so you can bootstrap a training set without manually labeling
photos. More (and more varied) real images will improve accuracy.
"""

import os

import cv2
import numpy as np

OUTPUT_DIR = "train_data"
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CANVAS_SIZE = 40


def render_char(char: str) -> np.ndarray:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (tw, th), base = cv2.getTextSize(char, font, scale, thickness)
    x = (CANVAS_SIZE - tw) // 2
    y = (CANVAS_SIZE + th) // 2
    cv2.putText(canvas, char, (x, y), font, scale, 255, thickness)
    return cv2.resize(canvas, (20, 20))


def generate(per_char: int = 20, out_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, char in enumerate(CHARS):
        char_dir = os.path.join(out_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        for j in range(per_char):
            img = render_char(char)
            rng = np.random.default_rng(i * 1000 + j)
            img = img + rng.integers(-25, 25, size=img.shape).astype(np.int8)
            img = np.clip(img, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(char_dir, f"{char}_{j:03d}.png"), img)
    print(f"Generated {len(CHARS)} characters x {per_char} images in {out_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-char", type=int, default=20)
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()
    generate(args.per_char, args.out)
