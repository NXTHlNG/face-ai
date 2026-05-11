from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.schemas.analysis import (
    AnalysisResponse,
    ColorFeatures,
    ConfidenceBreakdown,
    ContrastMetrics,
    GeometryMetrics,
    MetricsBundle,
    ParsingMeta,
    PhotoQuality,
    RecommendationItem,
)
from app.services.color_contrast import compute_contrast_and_color
from app.services import debug_dump
from app.services.face_parsing import glasses_pixel_ratio, parse_face
from app.services.geometry import compute_geometry
from app.services.landmarks import detect_landmarks
from app.services.photo_quality import assess_photo
from app.services.rules import build_recommendations


def _confidence(
    geometry: dict[str, Any],
    contrast: dict[str, Any],
    color: dict[str, Any],
    photo_ok: bool,
) -> ConfidenceBreakdown:
    notes: list[str] = []
    if not photo_ok:
        notes.append("photo_quality_reduces_confidence")

    g = float(geometry.get("face_shape_confidence", 0.4))
    if not contrast.get("parsing_used"):
        c_conf = 0.52
        notes.append("contrast_fallback_masks")
    else:
        c_conf = 0.78
    if contrast.get("iris_skin_delta_L") is not None and contrast.get("lip_skin_delta_L") is not None:
        c_conf = min(0.9, c_conf + 0.06)
    elif contrast.get("iris_skin_delta_L") is not None:
        c_conf = min(0.88, c_conf + 0.04)

    cf = float(color.get("seasonal_confidence", 0.35))
    color_conf = cf * (0.85 if contrast.get("parsing_used") else 0.65)

    overall = (
        g * 0.38 + c_conf * 0.32 + color_conf * 0.30
    ) * (0.65 if not photo_ok else 1.0)
    overall = max(0.0, min(1.0, overall))

    return ConfidenceBreakdown(
        overall=round(overall, 4),
        geometry=round(g * (0.85 if photo_ok else 0.55), 4),
        contrast=round(c_conf * (0.9 if photo_ok else 0.65), 4),
        color=round(color_conf * (0.9 if photo_ok else 0.65), 4),
        notes=notes,
    )


def analyze_image_bytes(data: bytes) -> AnalysisResponse:
    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        pq = PhotoQuality(
            sharpness_score=0.0,
            exposure_score=0.0,
            face_coverage_ratio=0.0,
            passes_gate=False,
            issues=["decode_failed"],
        )
        return AnalysisResponse(
            photo_quality=pq,
            metrics=None,
            confidence=ConfidenceBreakdown(
                overall=0.0,
                geometry=0.0,
                contrast=0.0,
                color=0.0,
                notes=["invalid_image"],
            ),
            recommendations=[],
        )
    return analyze_bgr(image_bgr)


def analyze_bgr(image_bgr: np.ndarray) -> AnalysisResponse:
    run_dir = debug_dump.new_run_dir()
    debug_dump.save_input(run_dir, image_bgr)

    if settings.landmark_backend == "dlib81":
        from app.services.dlib_landmarks import detect_landmarks_dlib81

        lm_result = detect_landmarks_dlib81(image_bgr)
    else:
        lm_result = detect_landmarks(image_bgr)
    bbox = lm_result.face_bbox_xywh if lm_result else None

    pq_dict, passes, issues = assess_photo(image_bgr, bbox)
    pq = PhotoQuality(
        sharpness_score=pq_dict["sharpness_score"],
        exposure_score=pq_dict["exposure_score"],
        face_coverage_ratio=pq_dict["face_coverage_ratio"],
        passes_gate=passes,
        issues=issues,
    )

    if lm_result is None:
        debug_dump.save_no_face(run_dir)
        return AnalysisResponse(
            photo_quality=pq,
            metrics=None,
            confidence=ConfidenceBreakdown(
                overall=0.0,
                geometry=0.0,
                contrast=0.0,
                color=0.0,
                notes=["no_face"],
            ),
            recommendations=[],
        )

    debug_dump.save_landmarks(run_dir, image_bgr, lm_result)

    if settings.skip_analysis_if_photo_poor and not passes:
        conf = _confidence(
            {"face_shape_confidence": 0.0},
            {"parsing_used": False},
            {"seasonal_confidence": 0.0},
            False,
        )
        debug_dump.save_gate_blocked(run_dir, issues)
        debug_dump.save_meta(
            run_dir,
            parsing_used=False,
            passes_gate=False,
            issues=issues,
            landmark_backend=settings.landmark_backend,
        )
        return AnalysisResponse(
            photo_quality=pq,
            metrics=None,
            confidence=conf,
            recommendations=[
                RecommendationItem(
                    category="general",
                    title="Повторите снимок",
                    detail=(
                        "Слишком размыто, плохая экспозиция или лицо слишком далеко. "
                        "Нужен крупный план при дневном свете, без сильного фильтра."
                    ),
                    rule_id="gate_retry_photo",
                    rule_version=settings.rules_version,
                    based_on={"issues": issues},
                )
            ],
        )

    pr = parse_face(lm_result.image_rgb, lm_result.landmarks_px)
    geometry = compute_geometry(lm_result)
    vendor_face_contour = geometry.pop("vendor_face_contour_px", None)
    g_ratio = glasses_pixel_ratio(pr)
    contrast, color = compute_contrast_and_color(
        lm_result.image_rgb,
        pr,
        lm_result.landmarks_px,
        g_ratio,
    )

    metrics = MetricsBundle(
        geometry=GeometryMetrics(**geometry),
        contrast=ContrastMetrics(**contrast),
        color_features=ColorFeatures(**color),
        parsing=ParsingMeta(
            used_model=pr.parsing_used,
            glasses_mask_ratio=g_ratio,
        ),
    )

    conf = _confidence(geometry, contrast, color, passes)
    raw_recs = build_recommendations(geometry, contrast, color, g_ratio)
    recs = [RecommendationItem(**r) for r in raw_recs]

    debug_dump.save_label_map_raw(run_dir, pr.label_map)
    debug_dump.save_face_contour_debug(run_dir, image_bgr, vendor_face_contour)
    debug_dump.save_masks(run_dir, pr)
    debug_dump.save_iris_ring_debug(
        run_dir,
        lm_result.image_rgb,
        lm_result.landmarks_px,
        pr,
        g_ratio,
    )
    debug_dump.save_parsing_overlay(run_dir, lm_result.image_rgb, pr)
    debug_dump.save_meta(
        run_dir,
        parsing_used=pr.parsing_used,
        passes_gate=True,
        issues=issues,
        landmark_backend=settings.landmark_backend,
    )

    return AnalysisResponse(
        photo_quality=pq,
        metrics=metrics,
        confidence=conf,
        recommendations=recs,
    )
