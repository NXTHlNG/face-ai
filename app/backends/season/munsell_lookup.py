from __future__ import annotations

import json
from pathlib import Path

from app.backends.season.aua_scales import (
    lab_chroma_to_aua,
    lab_luminance_contrast_aua,
    normalize_chroma_aua,
    normalize_contrast_aua,
    normalize_value_aua,
    opencv_L_to_aua_value,
)
from app.config import settings

_LOOKUP: dict | None = None


def _load() -> dict:
    global _LOOKUP
    if _LOOKUP is None:
        path = settings.data_dir / "season_lookup_16.json"
        _LOOKUP = json.loads(path.read_text(encoding="utf-8"))
    return _LOOKUP


def compute_munsell_axes(
    *,
    undertone: str,
    skin_ab: tuple[float, float],
    skin_L: float,
    eyes_L: float | None,
    hair_L: float | None,
    L_min: float,
    L_max: float,
    eyes_ab: tuple[float, float] | None = None,
    lips_ab: tuple[float, float] | None = None,
) -> dict[str, int]:
    from app.backends.color.aua_heuristics import aggregate_chroma

    ut = 5 if undertone == "warm" else (1 if undertone == "cool" else 3)

    chroma_lab = aggregate_chroma(skin_ab, eyes_ab, lips_ab)
    chroma_aua = lab_chroma_to_aua(chroma_lab)
    chroma = normalize_chroma_aua(chroma_aua)

    eyes = eyes_L if eyes_L is not None else skin_L
    hair = hair_L if hair_L is not None else skin_L - 12.0
    value_raw = 0.7 * skin_L + 0.15 * eyes + 0.15 * hair
    value_aua = opencv_L_to_aua_value(value_raw)
    value = normalize_value_aua(value_aua)

    contrast_aua = lab_luminance_contrast_aua(L_min, L_max)
    contrast = normalize_contrast_aua(contrast_aua)

    return {
        "undertone": ut,
        "chroma": chroma,
        "value": value,
        "contrast": contrast,
    }


def classify_munsell_sixteen(axes: dict[str, int]) -> dict:
    data = _load()
    best_id = "unknown"
    best_parent = "unknown"
    best_score = -1.0
    scores: dict[str, float] = {}

    for entry in data["types"]:
        tid = entry["id"]
        s = 0.0
        for axis in ("undertone", "chroma", "value", "contrast"):
            lo, hi = entry[axis]
            v = axes[axis]
            if lo <= v <= hi:
                s += 2.0
            else:
                dist = min(abs(v - lo), abs(v - hi))
                s += max(0.0, 2.0 - dist * 0.8)
        scores[tid] = s
        if s > best_score:
            best_score = s
            best_parent = entry["parent"]
            best_id = tid

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    total = sum(scores.values()) + 1e-9
    conf = float(min(0.85, max(0.25, best_score / (total / max(len(scores), 1)) * 0.15)))

    return {
        "seasonal_sixteen": best_id,
        "seasonal_guess": best_parent,
        "munsell_scores": axes,
        "confidence": conf,
        "scores": scores,
        "top_k_sixteen": [
            {"type": k, "probability": round(v / total, 4)} for k, v in ranked[:3]
        ],
    }
