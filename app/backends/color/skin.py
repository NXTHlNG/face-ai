from __future__ import annotations

import json

import cv2
import numpy as np

from app.backends.color.aua_heuristics import skin_lab_hue_trim
from app.backends.color.delta_e import nearest_palette_class, opencv_lab_to_cielab
from app.backends.color.lab_utils import mean_lab, median_lab
from app.config import settings


def _load_skin_palette() -> dict[str, tuple[float, float, float]]:
    raw = json.loads(
        (settings.data_dir / "reference_swatches.json").read_text(encoding="utf-8")
    )
    return {k: tuple(v) for k, v in raw["skin_tone_8"].items()}


def extract_mean_lab(image_rgb: np.ndarray, skin_mask: np.ndarray) -> dict:
    lab = skin_lab_hue_trim(image_rgb, skin_mask)
    if lab is None:
        lab = median_lab(image_rgb, skin_mask) or mean_lab(image_rgb, skin_mask)
    if lab is None:
        return {"lab": None, "ab": (0.0, 0.0), "method": "mean_lab"}
    ab = (float(lab[1]) - 128.0, float(lab[2]) - 128.0)
    return {"lab": lab, "ab": ab, "method": "mean_lab_hue_trim", "L": float(lab[0])}


def _kmeans_dominant(points: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = points.shape[0]
    if n <= k:
        return np.mean(points, axis=0)
    idx = rng.choice(n, size=k, replace=False)
    centers = points[idx].astype(np.float64)
    for _ in range(iters):
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for j in range(k):
            sel = points[labels == j]
            if sel.size:
                centers[j] = np.mean(sel, axis=0)
    counts = np.bincount(labels, minlength=k)
    return centers[int(np.argmax(counts))]


def extract_xmeans_hsv_deltae(image_rgb: np.ndarray, skin_mask: np.ndarray) -> dict:
    m = skin_mask > 127
    if np.count_nonzero(m) < 20:
        return extract_mean_lab(image_rgb, skin_mask)
    pixels = image_rgb[m]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    n_clusters = min(4, max(2, len(pixels) // 80))
    dom_hsv = _kmeans_dominant(hsv.astype(np.float32), n_clusters)
    dom_rgb = cv2.cvtColor(
        np.uint8([[dom_hsv]]), cv2.COLOR_HSV2RGB
    )[0, 0]
    cielab = opencv_lab_to_cielab(
        *cv2.cvtColor(np.uint8([[dom_rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    )
    palette = _load_skin_palette()
    tone_class, dist, scores = nearest_palette_class(cielab, palette)
    opencv_lab = cv2.cvtColor(
        np.uint8([[dom_rgb]]), cv2.COLOR_RGB2LAB
    )[0, 0].astype(float)
    return {
        "lab": tuple(opencv_lab),
        "ab": (float(opencv_lab[1]) - 128.0, float(opencv_lab[2]) - 128.0),
        "method": "xmeans_hsv_deltae",
        "L": float(opencv_lab[0]),
        "skin_tone_class": tone_class,
        "delta_e_scores": {k: round(v, 3) for k, v in scores.items()},
    }


def extract_skin(image_rgb: np.ndarray, skin_mask: np.ndarray) -> dict:
    if settings.skin_color_backend == "xmeans_hsv_deltae":
        return extract_xmeans_hsv_deltae(image_rgb, skin_mask)
    return extract_mean_lab(image_rgb, skin_mask)
