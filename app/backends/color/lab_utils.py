from __future__ import annotations

import cv2
import numpy as np


def mean_lab(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    m = mask > 127
    if not np.any(m):
        return None
    roi = image_rgb[m].reshape(-1, 3)
    if roi.size == 0:
        return None
    lab = cv2.cvtColor(roi.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return tuple(float(np.mean(lab[:, i])) for i in range(3))


def median_lab(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    m = mask > 127
    if not np.any(m):
        return None
    roi = image_rgb[m].reshape(-1, 3)
    if roi.size == 0:
        return None
    lab = cv2.cvtColor(roi.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return tuple(float(np.median(lab[:, i])) for i in range(3))


def ab_from_opencv_lab(lab: tuple[float, float, float]) -> tuple[float, float]:
    return (float(lab[1]) - 128.0, float(lab[2]) - 128.0)
