from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParsingResult:
    skin_mask: np.ndarray
    hair_mask: np.ndarray
    brow_mask: np.ndarray
    eye_glass_mask: np.ndarray | None
    lip_mask: np.ndarray | None
    eye_region_mask: np.ndarray | None
    parsing_used: bool
    label_map: np.ndarray | None
    parsing_backend: str = "landmark_fallback"
    features_union_mask: np.ndarray | None = None


def glasses_pixel_ratio(pr: ParsingResult) -> float | None:
    if pr.eye_glass_mask is None:
        return None
    return float(np.mean(pr.eye_glass_mask > 127))
