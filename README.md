# License Plate Recognition with Machine Learning in Python

A classic Automatic License Plate Recognition (ALPR) pipeline, based on the
tutorial "Developing a License Plate Recognition System with Machine Learning
in Python". It detects a license plate in a photo, segments the characters,
and reads them with a machine-learned classifier.

## Pipeline

The system is split into three reusable stages:

1. **Localization** (`lpr/localization.py`)
   Converts to grayscale, denoises with a bilateral filter, finds strong
   edges with a Canny/Sobel operator, closes gaps with morphological
   dilation, and keeps the four-cornered contour that looks like a plate.
   The plate is then perspective-corrected into a clean crop.

2. **Segmentation** (`lpr/segmentation.py`)
   Thresholds the plate and finds each character with connected components,
   filtering candidates by shape and size.

3. **Recognition** (`lpr/recognition.py`)
   Each 20x20 character image is flattened and classified with an SVM (RBF
   kernel) trained on labeled character images — the machine learning core.

## Project structure

```
license-plate-recognition/
├── lpr/
│   ├── __init__.py
│   ├── localization.py    # find + correct the plate
│   ├── segmentation.py    # cut the plate into characters
│   ├── recognition.py     # SVM character classifier
│   └── cli.py             # command-line pipeline
├── generate_training_data.py  # synthesize labeled chars for training
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Requires a Tesseract-free setup — OCR is done entirely with the trained SVM,
so no external binary dependencies.

## Getting started

### 1. Train the character recognizer

First bootstrap a model. You can synthesize a labeled dataset (rendered font
characters) and train on it:

```bash
python generate_training_data.py --per-char 20 --out train_data
python -m lpr.cli train train_data
```

For better accuracy, drop real cropped 20x20 character PNGs into per-character
folders (e.g. `train_data/A/`, `train_data/7/`, ...) and retrain.

### 2. Recognize a plate

```bash
python -m lpr.cli detect cars/car1.jpg
```

Add `--show` to display the image with the detected plate outlined.

## Example

```bash
$ python -m lpr.cli detect sample_car.jpg
License plate: ABC123
```

## Notes

- Accuracy depends heavily on training data diversity and image quality;
  real-world plates are harder than synthetic ones. Synthetic training is
  enough to demonstrate the full pipeline end to end.
- The SVM is stored at `models/char_svm.pkl` (ignored by git).
