"""Character segmentation: split the localized plate into individual characters.

The plate crop is thresholded and characters are found via connected
components (contours). Candidates whose dimensions and position match the
typical shape of a character are kept, so stray marks around the plate are
filtered out.
"""

from typing import List

import cv2
import numpy as np


def preprocess_plate(plate: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 17, 17)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def segment_characters(plate: np.ndarray) -> List[np.ndarray]:
    """Return a list of resized, cropped character images from the plate.

    Each character is 20x20 pixels, ordered left-to-right as they appear on
    the plate. Filtering is scale-adaptive: size thresholds are derived from
    the plate's character height so the same code works on any plate
    resolution.
    """
    plate_h, plate_w = plate.shape[:2]
    char_height = plate_h * 0.75

    thresh = preprocess_plate(plate)
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > plate_w * 0.7:
            continue
        if h < char_height * 0.35 or h > plate_h:
            continue
        aspect_ratio = h / float(w)
        if 0.25 < aspect_ratio < 2.2:
            candidates.append((x, y, w, h, thresh[y : y + h, x : x + w]))

    candidates.sort(key=lambda c: c[0])

    characters = []
    for _, _, _, _, candidate in candidates:
        resized = cv2.resize(candidate, (20, 20))
        if resized.shape[0] > 0 and resized.shape[1] > 0:
            characters.append(resized)

    return characters
