"""End-to-end license plate recognition pipeline (CLI)."""

import argparse
import os

import cv2

from lpr import localize_plate, segment_characters, CharacterRecognizer
from lpr.recognition import MODEL_PATH as MODEL_DEFAULT


def recognize_plate(image_path: str, show: bool = False) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    plate, contour = localize_plate(image)
    if plate is None:
        return ""

    characters = segment_characters(plate)
    if not characters:
        return ""

    recognizer = CharacterRecognizer()
    if not recognizer.trained:
        raise RuntimeError(
            "No trained character model found. Train one first with --train."
        )

    predictions = recognizer.predict(characters)
    text = "".join(predictions)

    if show and contour is not None:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
        cv2.imshow("Detected plate", cv2.resize(image, None, fx=2, fy=2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return text


def train_recognizer(data_dir: str, model_path: str) -> dict:
    """Train the character recognizer on a folder of labeled character images.

    The data directory should contain one subfolder per character (e.g.
    'A/', 'B/', '0/', '1/'), each holding 20x20 grayscale PNG images of that
    character.
    """
    images, labels = [], []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for name in os.listdir(label_dir):
            img = cv2.imread(os.path.join(label_dir, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (20, 20))
            images.append(img)
            labels.append(label)

    if not images:
        raise ValueError(f"No training images found under {data_dir}")

    recognizer = CharacterRecognizer(model_path=model_path)
    return recognizer.train(images, labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="License plate recognition with machine learning in Python"
    )
    sub = parser.add_subparsers(dest="command")

    detect = sub.add_parser("detect", help="Recognize the plate in an image")
    detect.add_argument("image", help="Path to the input image")
    detect.add_argument("--show", action="store_true", help="Display the detected plate")

    train = sub.add_parser("train", help="Train the character recognition model")
    train.add_argument("data_dir", help="Directory of labeled character images")
    train.add_argument("--model", default=None, help="Where to save the model")

    args = parser.parse_args()

    if args.command == "detect":
        try:
            text = recognize_plate(args.image, show=args.show)
            print(f"License plate: {text}" if text else "No plate detected.")
        except RuntimeError as exc:
            print(f"Error: {exc}")
    elif args.command == "train":
        model = args.model or MODEL_DEFAULT
        scores = train_recognizer(args.data_dir, model)
        print(f"Trained model saved to {model}")
        for k, v in scores.items():
            print(f"  {k}: {v:.3f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
