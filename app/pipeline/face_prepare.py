from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.backends.parsing import parse_face
from app.backends.parsing.registry import ParsingBackendId
from app.backends.parsing.types import ParsingResult
from app.config import settings
from app.pipeline.mask_postprocess import enhance_parsing
from app.services.landmarks import detect_landmarks
from app.services.photo_quality import assess_photo


@dataclass
class PreparedFace:
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    landmarks_px: np.ndarray
    parsing: ParsingResult
    photo_passes_gate: bool
    photo_issues: list[str]


def prepare_face_bgr(
    image_bgr: np.ndarray,
    *,
    parsing_backend: ParsingBackendId | None = None,
    skip_quality_gate: bool = True,
) -> PreparedFace | None:
    if settings.landmark_backend == "dlib81":
        from app.services.dlib_landmarks import detect_landmarks_dlib81

        lm_result = detect_landmarks_dlib81(image_bgr)
    else:
        lm_result = detect_landmarks(image_bgr)

    if lm_result is None:
        return None

    bbox = lm_result.face_bbox_xywh
    _, passes, issues = assess_photo(image_bgr, bbox)
    if not skip_quality_gate and settings.skip_analysis_if_photo_poor and not passes:
        return None

    pr, _, _ = parse_face(
        lm_result.image_rgb,
        lm_result.landmarks_px,
        backend=parsing_backend,
    )
    img_work, pr = enhance_parsing(
        lm_result.image_rgb,
        pr,
        lm_result.landmarks_px,
        exposure_score=1.0,
    )
    return PreparedFace(
        image_bgr=image_bgr,
        image_rgb=img_work,
        landmarks_px=lm_result.landmarks_px,
        parsing=pr,
        photo_passes_gate=passes,
        photo_issues=issues,
    )
