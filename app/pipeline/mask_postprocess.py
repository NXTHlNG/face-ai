from __future__ import annotations

import cv2
import numpy as np

from app.backends.parsing.types import ParsingResult
from app.config import settings
from app.services.mask_geometry import lip_mask_from_landmarks

# DL parsers already split classes; jaw hull + subtracting brow/lip erodes valid skin.
_DL_PARSING_BACKENDS = frozenset({"bisenet_resnet34", "farl_b", "segface"})


def _morph_clean(mask: np.ndarray) -> np.ndarray:
    m = (mask > 127).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def _gamma_correct(image_rgb: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 0.02:
        return image_rgb
    inv = 1.0 / max(gamma, 0.05)
    table = np.array(
        [((i / 255.0) ** inv) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image_rgb, table)


def cheek_jaw_skin_mask(
    image_shape: tuple[int, int],
    landmarks_px: np.ndarray,
) -> np.ndarray | None:
    h, w = image_shape[:2]
    if landmarks_px.shape[0] >= 81:
        jaw_idx = list(range(0, 17))
        cheek_idx = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        pts = []
        for i in jaw_idx + cheek_idx:
            if i < len(landmarks_px):
                pts.append([float(landmarks_px[i, 0]), float(landmarks_px[i, 1])])
    elif landmarks_px.shape[0] >= 68:
        jaw_idx = list(range(0, 17))
        pts = [[float(landmarks_px[i, 0]), float(landmarks_px[i, 1])] for i in jaw_idx]
    else:
        return None
    if len(pts) < 4:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    arr = np.asarray(pts, dtype=np.float32)
    hull = cv2.convexHull(arr)
    poly = np.round(hull).astype(np.int32).reshape(-1, 2)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def refine_skin_mask(
    skin_mask: np.ndarray,
    *,
    cheek_jaw: np.ndarray | None = None,
    lip_mask: np.ndarray | None = None,
    brow_mask: np.ndarray | None = None,
) -> np.ndarray:
    skin = _morph_clean(skin_mask)
    if cheek_jaw is not None and np.any(cheek_jaw > 127):
        skin = cv2.bitwise_and(skin, cheek_jaw)
    if lip_mask is not None:
        skin = cv2.bitwise_and(skin, cv2.bitwise_not(lip_mask))
    if brow_mask is not None:
        skin = cv2.bitwise_and(skin, cv2.bitwise_not(brow_mask))
    return skin


def blur_skin_roi(image_rgb: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    sigma = max(1.0, min(h, w) * settings.skin_blur_sigma_ratio)
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    m = skin_mask > 127
    out = image_rgb.copy()
    out[m] = blurred[m]
    return out


def _resolve_lip_mask(
    lip_mask: np.ndarray | None,
    image_shape: tuple[int, ...],
    landmarks_px: np.ndarray,
) -> np.ndarray | None:
    lip = lip_mask
    min_px = max(400, int(image_shape[0] * image_shape[1] * 0.00015))
    lip_px = int(np.count_nonzero(lip > 127)) if lip is not None else 0
    if lip_px >= min_px:
        return lip
    lm_lip = lip_mask_from_landmarks(image_shape, landmarks_px)
    if lm_lip is not None and np.any(lm_lip > 127):
        return lm_lip
    return lip


def enhance_parsing(
    image_rgb: np.ndarray,
    pr: ParsingResult,
    landmarks_px: np.ndarray,
    *,
    exposure_score: float = 1.0,
    apply_skin_blur: bool = True,
) -> tuple[np.ndarray, ParsingResult]:
    gamma = settings.gamma_correction
    if exposure_score < 0.45:
        gamma = max(0.85, gamma * 0.92)
    img = _gamma_correct(image_rgb, gamma)

    use_geometry = (
        pr.parsing_backend not in _DL_PARSING_BACKENDS or not pr.parsing_used
    )
    cheek = (
        cheek_jaw_skin_mask(img.shape, landmarks_px)
        if use_geometry and landmarks_px.shape[0] >= 68
        else None
    )
    skin = refine_skin_mask(
        pr.skin_mask,
        cheek_jaw=cheek,
        lip_mask=pr.lip_mask,
        brow_mask=pr.brow_mask,
    )
    hair = _morph_clean(pr.hair_mask)
    brow = _morph_clean(pr.brow_mask)
    lip_raw = _resolve_lip_mask(pr.lip_mask, img.shape, landmarks_px)
    lip = _morph_clean(lip_raw) if lip_raw is not None else lip_raw
    eye = _morph_clean(pr.eye_region_mask) if pr.eye_region_mask is not None else pr.eye_region_mask

    union = np.clip(
        skin.astype(np.float32)
        + hair.astype(np.float32)
        + (brow.astype(np.float32) if brow is not None else 0)
        + (eye.astype(np.float32) if eye is not None else 0),
        0,
        255,
    ).astype(np.uint8)

    new_pr = ParsingResult(
        skin_mask=skin,
        hair_mask=hair,
        brow_mask=brow,
        eye_glass_mask=pr.eye_glass_mask,
        lip_mask=lip,
        eye_region_mask=eye,
        parsing_used=pr.parsing_used,
        label_map=pr.label_map,
        parsing_backend=pr.parsing_backend,
        features_union_mask=union,
    )
    img_out = blur_skin_roi(img, skin) if apply_skin_blur else img
    return img_out, new_pr
