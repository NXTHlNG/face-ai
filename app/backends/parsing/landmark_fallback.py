from __future__ import annotations

import cv2
import numpy as np

from app.backends.parsing.types import ParsingResult

FACE_OVAL_IDX = np.array(
    [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109,
    ],
    dtype=np.int32,
)
LEFT_BROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_IDX = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
DLIB_LEFT_BROW = list(range(17, 22))
DLIB_RIGHT_BROW = list(range(22, 27))


def _fill_convex_poly_int32(mask: np.ndarray, pts: list[list[float]]) -> None:
    if len(pts) < 3:
        return
    h, w = mask.shape[:2]
    arr = np.asarray(pts, dtype=np.float32)
    hull = cv2.convexHull(arr)
    poly = np.round(hull).astype(np.int32).reshape(-1, 2)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    if poly.shape[0] < 3:
        return
    cv2.fillConvexPoly(mask, poly, 255)


def parse(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
) -> ParsingResult:
    if landmarks_px.shape[0] == 81:
        return _dlib81(image_rgb.shape, landmarks_px)
    return _mediapipe(image_rgb.shape, landmarks_px)


def _dlib81(shape: tuple, landmarks_px: np.ndarray) -> ParsingResult:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_all = [
        [float(landmarks_px[i, 0]), float(landmarks_px[i, 1])]
        for i in range(len(landmarks_px))
    ]
    _fill_convex_poly_int32(mask, pts_all)
    brow = np.zeros_like(mask)
    for idx_list in (DLIB_LEFT_BROW, DLIB_RIGHT_BROW):
        bpts = [[float(landmarks_px[i, 0]), float(landmarks_px[i, 1])] for i in idx_list]
        _fill_convex_poly_int32(brow, bpts)
    hair = np.zeros_like(mask)
    top_y = int(np.clip(np.min(landmarks_px[:, 1]) - (h * 0.08), 0, h - 1))
    cv2.rectangle(hair, (0, 0), (w - 1, top_y + max(1, h // 25)), 255, thickness=-1)
    zones = {
        "skin": mask,
        "hair": hair,
        "brow": brow,
        "glasses": np.zeros_like(mask),
        "lip": np.zeros_like(mask),
        "eye": np.zeros_like(mask),
    }
    from app.backends.parsing.label_maps import canonical_to_parsing_result

    return canonical_to_parsing_result(
        zones,
        parsing_used=False,
        label_map=None,
        parsing_backend="landmark_fallback",
    )


def _mediapipe(shape: tuple, landmarks_px: np.ndarray) -> ParsingResult:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = []
    for i in FACE_OVAL_IDX:
        if i < len(landmarks_px):
            pts.append([landmarks_px[i, 0], landmarks_px[i, 1]])
    _fill_convex_poly_int32(mask, pts)
    brow = np.zeros_like(mask)
    for idx_list in (LEFT_BROW_IDX, RIGHT_BROW_IDX):
        bpts = []
        for i in idx_list:
            if i < len(landmarks_px):
                bpts.append([landmarks_px[i, 0], landmarks_px[i, 1]])
        _fill_convex_poly_int32(brow, bpts)
    hair = np.zeros_like(mask)
    top_y = int(np.clip(np.min(landmarks_px[:, 1]) - (h * 0.08), 0, h - 1))
    cv2.rectangle(hair, (0, 0), (w - 1, top_y + max(1, h // 25)), 255, thickness=-1)
    zones = {
        "skin": mask,
        "hair": hair,
        "brow": brow,
        "glasses": np.zeros_like(mask),
        "lip": np.zeros_like(mask),
        "eye": np.zeros_like(mask),
    }
    from app.backends.parsing.label_maps import canonical_to_parsing_result

    return canonical_to_parsing_result(
        zones,
        parsing_used=False,
        label_map=None,
        parsing_backend="landmark_fallback",
    )
