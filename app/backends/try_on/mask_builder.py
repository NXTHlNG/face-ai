from __future__ import annotations

import cv2
import numpy as np

from app.backends.parsing.types import ParsingResult
from app.pipeline.mask_postprocess import cheek_jaw_skin_mask
from app.services.mask_geometry import lip_mask_from_landmarks

_MP_GLASSES_IDX = (
    33,
    133,
    160,
    159,
    158,
    144,
    145,
    153,
    362,
    263,
    387,
    386,
    385,
    373,
    374,
    380,
    168,
    6,
    127,
    356,
    234,
    454,
)


def _empty_mask(shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def _feather_mask(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if int(np.max(mask)) < 1:
        return mask
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _union_masks(*masks: np.ndarray | None) -> np.ndarray:
    if not masks:
        raise ValueError("no masks")
    base = None
    for m in masks:
        if m is None:
            continue
        if base is None:
            base = (m > 127).astype(np.uint8) * 255
        else:
            base = np.maximum(base, (m > 127).astype(np.uint8) * 255)
    if base is None:
        return _empty_mask(masks[0].shape if masks[0] is not None else (1, 1))
    return base


def _subtract_mask(base: np.ndarray, *others: np.ndarray | None) -> np.ndarray:
    out = (base > 127).astype(np.uint8)
    for o in others:
        if o is not None:
            out = out & ~(o > 127)
    return out.astype(np.uint8) * 255


def build_hairstyle_mask(
    pr: ParsingResult,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    hair = pr.hair_mask
    if hair is None or int(np.max(hair)) < 1:
        return _empty_mask(image_shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hair = cv2.morphologyEx((hair > 127).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
    return _feather_mask(hair, sigma=4.0)


def build_makeup_mask(
    pr: ParsingResult,
    image_shape: tuple[int, ...],
    landmarks_px: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    h, w = image_shape[:2]
    zones: list[str] = []
    parts: list[np.ndarray] = []

    lip = pr.lip_mask
    if lip is None or int(np.max(lip)) < 1:
        if landmarks_px is not None:
            lip = lip_mask_from_landmarks(image_shape, landmarks_px)
    if lip is not None and int(np.max(lip)) > 0:
        parts.append(lip)
        zones.append("lips")

    if pr.brow_mask is not None and int(np.max(pr.brow_mask)) > 0:
        parts.append(pr.brow_mask)
        zones.append("brows")

    skin = pr.skin_mask
    cheek = (
        cheek_jaw_skin_mask(image_shape, landmarks_px)
        if landmarks_px is not None and landmarks_px.shape[0] >= 68
        else None
    )
    if cheek is not None and skin is not None:
        blush = _subtract_mask(
            np.minimum(cheek, skin),
            lip,
            pr.eye_region_mask,
            pr.brow_mask,
        )
        if int(np.max(blush)) > 0:
            parts.append(blush)
            zones.append("blush")

    if pr.eye_region_mask is not None and int(np.max(pr.eye_region_mask)) > 0:
        shadow = pr.eye_region_mask.copy()
        if shadow.ndim == 3:
            shadow = cv2.cvtColor(shadow, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        inner = cv2.erode((shadow > 127).astype(np.uint8) * 255, k, iterations=2)
        shadow = _subtract_mask(shadow, inner, pr.brow_mask)
        if int(np.max(shadow)) > 0:
            parts.append(shadow)
            zones.append("shadow")

    if not parts:
        return _empty_mask(image_shape), zones
    union = _union_masks(*parts)
    return _feather_mask(union, sigma=2.5), zones


def build_glasses_mask(
    pr: ParsingResult,
    image_shape: tuple[int, ...],
    landmarks_px: np.ndarray | None,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = _empty_mask(image_shape)

    if landmarks_px is not None and landmarks_px.shape[0] >= 468:
        pts = []
        for i in _MP_GLASSES_IDX:
            if i < len(landmarks_px):
                pts.append([float(landmarks_px[i, 0]), float(landmarks_px[i, 1])])
        if len(pts) >= 4:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad_x = 0.12 * (max(xs) - min(xs) + 1)
            pad_y = 0.25 * (max(ys) - min(ys) + 1)
            x0 = int(max(0, min(xs) - pad_x))
            x1 = int(min(w - 1, max(xs) + pad_x))
            y0 = int(max(0, min(ys) - pad_y))
            y1 = int(min(h - 1, max(ys) + pad_y))
            mask[y0 : y1 + 1, x0 : x1 + 1] = 255

    if pr.eye_region_mask is not None and int(np.max(pr.eye_region_mask)) > 0:
        eye = pr.eye_region_mask
        if eye.shape[:2] != (h, w):
            eye = cv2.resize(eye, (w, h), interpolation=cv2.INTER_NEAREST)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eye = cv2.dilate((eye > 127).astype(np.uint8) * 255, k, iterations=1)
        mask = np.maximum(mask, eye)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate((mask > 127).astype(np.uint8) * 255, k, iterations=1)
    return _feather_mask(mask, sigma=2.0)
