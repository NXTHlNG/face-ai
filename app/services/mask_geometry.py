"""Геометрические маски по landmarks (fallback, когда parsing даёт слишком маленькую зону)."""

from __future__ import annotations

import cv2
import numpy as np

_MP_OUTER_LIP_RING = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
)


def lip_mask_from_landmarks(image_shape: tuple[int, ...], landmarks_px: np.ndarray) -> np.ndarray | None:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks_px.shape[0] >= 478:
        pts = []
        for i in _MP_OUTER_LIP_RING:
            if i < len(landmarks_px):
                pts.append([float(landmarks_px[i, 0]), float(landmarks_px[i, 1])])
        if len(pts) >= 3:
            arr = np.asarray(pts, dtype=np.float32)
            hull = cv2.convexHull(arr)
            poly = np.round(hull).astype(np.int32).reshape(-1, 2)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            cv2.fillConvexPoly(mask, poly, 255)
            return mask
    if landmarks_px.shape[0] >= 68:
        pts = [[float(landmarks_px[i, 0]), float(landmarks_px[i, 1])] for i in range(48, 60)]
        arr = np.asarray(pts, dtype=np.float32)
        hull = cv2.convexHull(arr)
        poly = np.round(hull).astype(np.int32).reshape(-1, 2)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        cv2.fillConvexPoly(mask, poly, 255)
        return mask
    return None
