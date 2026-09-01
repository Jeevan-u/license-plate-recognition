"""Plate localization: find the license plate region in an image.

The pipeline is the classic segmentation-based approach described in the
tutorial: convert to grayscale, remove noise with a bilateral filter, find
vertical edges with a Sobel operator, boost them with morphological
dilation, threshold to a binary mask, and keep the connected component that
most looks like a plate (a large, slightly-wide rectangle).
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def localize_plate(
    image: np.ndarray,
    morph_kernel_width: int = 17,
    morph_kernel_height: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return the cropped, perspective-corrected plate and the plate contour.

    Returns (plate, contour); if no plate is found both are None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    edged = cv2.Canny(gray, 30, 200)

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_width, morph_kernel_height)
    )
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, morph_kernel)
    closed = cv2.dilate(closed, None, iterations=1)

    contours, _ = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        return None, None

    plate = _four_point_transform(image, plate_contour.reshape(4, 2))
    return plate, plate_contour
