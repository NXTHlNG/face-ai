from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.backends.parsing import glasses_pixel_ratio, parse_face, parsing_health
from app.backends.parsing.registry import ParsingBackendId
from app.backends.color.extract import compute_contrast_and_color
from app.backends.wrist.undertone import analyze_wrist_undertone
from app.config import settings
from app.pipeline.mask_postprocess import enhance_parsing
from app.pipeline.mask_preview import build_mask_preview
from app.schemas.analysis import (
    AnalysisPalette,
    AnalysisResponse,
    BackendOverrides,
    ColorFeatures,
    ConfidenceBreakdown,
    ContrastMetrics,
    DebugBundle,
    GeometryMetrics,
    MaskPreview,
    MetricsBundle,
    PaletteSwatch,
    ParsingMeta,
    PhotoQuality,
    RecommendationItem,
    SeasonalAnalysis,
)
from app.services import debug_dump
from app.services.geometry import compute_geometry
from app.services.landmarks import detect_landmarks
from app.services.photo_quality import assess_photo
from app.backends.llm.adapter import LLMAnalysisAdapter
from app.backends.llm.schema_merge import apply_llm_seasonal
from app.services.rules import build_recommendations

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeOptions:
    parsing_backend: ParsingBackendId | None = None
    wrist_bgr: np.ndarray | None = None
    backends: BackendOverrides | None = None


def _confidence(
    geometry: dict[str, Any],
    contrast: dict[str, Any],
    seasonal: dict[str, Any],
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

    cf = float(seasonal.get("seasonal_confidence", 0.35))
    color_conf = cf * (0.85 if contrast.get("parsing_used") else 0.65)
    overall = (g * 0.38 + c_conf * 0.32 + color_conf * 0.30) * (0.65 if not photo_ok else 1.0)
    overall = max(0.0, min(1.0, overall))

    return ConfidenceBreakdown(
        overall=round(overall, 4),
        geometry=round(g * (0.85 if photo_ok else 0.55), 4),
        contrast=round(c_conf * (0.9 if photo_ok else 0.65), 4),
        color=round(color_conf * (0.9 if photo_ok else 0.65), 4),
        notes=notes + seasonal.get("seasonal_notes", []),
    )


def _resolve_parsing_backend(opts: AnalyzeOptions) -> ParsingBackendId | None:
    if opts.parsing_backend:
        return opts.parsing_backend
    if opts.backends and opts.backends.parsing_backend:
        return opts.backends.parsing_backend
    return None


def _attach_debug(resp: AnalysisResponse, run_dir) -> AnalysisResponse:
    bundle = debug_dump.build_debug_bundle(run_dir)
    if bundle is None:
        return resp
    return resp.model_copy(update={"debug": DebugBundle(**bundle)})


def analyze_bgr(image_bgr: np.ndarray, opts: AnalyzeOptions | None = None) -> AnalysisResponse:
    opts = opts or AnalyzeOptions()
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
        return _attach_debug(
            AnalysisResponse(
                photo_quality=pq,
                metrics=None,
                confidence=ConfidenceBreakdown(
                    overall=0.0, geometry=0.0, contrast=0.0, color=0.0, notes=["no_face"]
                ),
                recommendations=[],
            ),
            run_dir,
        )

    debug_dump.save_landmarks(run_dir, image_bgr, lm_result)

    if settings.skip_analysis_if_photo_poor and not passes:
        debug_dump.save_gate_blocked(run_dir, issues)
        return _attach_debug(
            AnalysisResponse(
                photo_quality=pq,
                metrics=None,
                confidence=_confidence(
                    {"face_shape_confidence": 0.0},
                    {"parsing_used": False},
                    {"seasonal_confidence": 0.0},
                    False,
                ),
                recommendations=[
                    RecommendationItem(
                        category="general",
                        title="Повторите снимок",
                        detail="Слишком размыто, плохая экспозиция или лицо слишком далеко.",
                        rule_id="gate_retry_photo",
                        rule_version=settings.rules_version,
                        based_on={"issues": issues},
                    )
                ],
            ),
            run_dir,
        )

    pb = _resolve_parsing_backend(opts)
    pr, _, parsing_notes = parse_face(
        lm_result.image_rgb,
        lm_result.landmarks_px,
        backend=pb,
    )
    debug_dump.save_masks(run_dir, pr, suffix="_parsing")
    img_work, pr = enhance_parsing(
        lm_result.image_rgb,
        pr,
        lm_result.landmarks_px,
        exposure_score=pq.exposure_score,
    )

    wrist_ut = None
    if opts.wrist_bgr is not None:
        wrist_ut = analyze_wrist_undertone(opts.wrist_bgr)

    g_ratio = glasses_pixel_ratio(pr)

    _orig_classifier = settings.season_classifier
    _orig_skin = settings.skin_color_backend
    if opts.backends:
        if opts.backends.season_classifier:
            settings.season_classifier = opts.backends.season_classifier
        if opts.backends.skin_color_backend:
            settings.skin_color_backend = opts.backends.skin_color_backend

    contrast, color, seasonal, analysis_palette = compute_contrast_and_color(
        img_work,
        pr,
        lm_result.landmarks_px,
        g_ratio,
        wrist_undertone=wrist_ut,
        exposure_score=pq.exposure_score,
    )
    settings.season_classifier = _orig_classifier
    settings.skin_color_backend = _orig_skin

    geometry = compute_geometry(lm_result)
    vendor_face_contour = geometry.pop("vendor_face_contour_px", None)

    mask_preview_dict = None
    if settings.mask_preview_enabled:
        mask_preview_dict = build_mask_preview(lm_result.image_rgb, pr)

    seasonal_model = SeasonalAnalysis(
        seasonal_guess=seasonal["seasonal_guess"],
        seasonal_twelve=seasonal["seasonal_twelve"],
        seasonal_sixteen=seasonal.get("seasonal_sixteen", "unknown"),
        seasonal_confidence=seasonal["seasonal_confidence"],
        seasonal_twelve_confidence=seasonal["seasonal_twelve_confidence"],
        seasonal_guess_top_k=seasonal.get("seasonal_guess_top_k", []),
        seasonal_twelve_top_k=seasonal.get("seasonal_twelve_top_k", []),
        munsell_scores=seasonal.get("munsell_scores"),
        skin_tone_class=color.get("skin_tone_class"),
        undertone_hint=color["undertone_hint"],
        undertone_source=color.get("undertone_source", "face"),
        classifier_contributors=seasonal.get("classifier_contributors", {}),
        delta_e_scores=color.get("delta_e_scores"),
        seasonal_method=seasonal.get("seasonal_method", "ensemble"),
    )

    palette_model = AnalysisPalette(
        face=[PaletteSwatch(**s) for s in analysis_palette.get("face", [])],
        season_reference=[
            PaletteSwatch(**s) for s in analysis_palette["season_reference"]
        ]
        if analysis_palette.get("season_reference")
        else None,
    )

    metrics = MetricsBundle(
        geometry=GeometryMetrics(**geometry),
        contrast=ContrastMetrics(**contrast),
        color_features=ColorFeatures(**{k: v for k, v in color.items() if k in ColorFeatures.model_fields}),
        parsing=ParsingMeta(
            used_model=pr.parsing_used,
            parsing_backend=pr.parsing_backend,
            glasses_mask_ratio=g_ratio,
        ),
        seasonal=seasonal_model,
        analysis_palette=palette_model,
        mask_preview=MaskPreview(**mask_preview_dict) if mask_preview_dict else None,
    )

    conf = _confidence(geometry, contrast, seasonal, passes)
    if parsing_notes:
        conf = conf.model_copy(update={"notes": conf.notes + parsing_notes})

    llm_used = False
    if settings.llm_available:
        logger.info(
            "LLM enabled; CV season before merge: guess=%s twelve=%s sixteen=%s",
            seasonal_model.seasonal_guess,
            seasonal_model.seasonal_twelve,
            seasonal_model.seasonal_sixteen,
        )
        llm_payload = LLMAnalysisAdapter().analyze(image_bgr)
        if llm_payload is None:
            logger.warning("LLM returned no payload")
        seasonal_model, llm_used = apply_llm_seasonal(
            seasonal_model,
            llm_payload.model_dump() if llm_payload else None,
        )
        if llm_used:
            logger.info(
                "LLM season applied: guess=%s twelve=%s sixteen=%s confidence=%.2f undertone=%s",
                seasonal_model.seasonal_guess,
                seasonal_model.seasonal_twelve,
                seasonal_model.seasonal_sixteen,
                seasonal_model.seasonal_confidence,
                seasonal_model.undertone_hint,
            )
            metrics = metrics.model_copy(update={"seasonal": seasonal_model})
            seasonal = {
                **seasonal,
                "seasonal_guess": seasonal_model.seasonal_guess,
                "seasonal_twelve": seasonal_model.seasonal_twelve,
                "seasonal_sixteen": seasonal_model.seasonal_sixteen,
                "seasonal_confidence": seasonal_model.seasonal_confidence,
                "seasonal_twelve_confidence": seasonal_model.seasonal_twelve_confidence,
            }
            color = {
                **color,
                "seasonal_guess": seasonal_model.seasonal_guess,
                "seasonal_twelve": seasonal_model.seasonal_twelve,
                "undertone_hint": seasonal_model.undertone_hint,
            }
        else:
            logger.warning("LLM call failed or invalid JSON; keeping CV/rules season")
    else:
        logger.info(
            "LLM skipped (enabled=%s, url_set=%s); using CV/rules season",
            settings.llm_enabled,
            bool(settings.llm_api_url.strip()),
        )

    raw_recs = build_recommendations(
        geometry, contrast, color, g_ratio, seasonal.get("seasonal_sixteen")
    )
    recs = [RecommendationItem(**r) for r in raw_recs]

    debug_dump.save_label_map_raw(run_dir, pr.label_map)
    debug_dump.save_face_contour_debug(run_dir, image_bgr, vendor_face_contour)
    debug_dump.save_masks(run_dir, pr)
    debug_dump.save_iris_ring_debug(run_dir, lm_result.image_rgb, lm_result.landmarks_px, pr, g_ratio)
    debug_dump.save_parsing_overlay(run_dir, lm_result.image_rgb, pr)
    debug_dump.save_seasonal_debug(run_dir, seasonal)
    debug_dump.save_meta(
        run_dir,
        parsing_used=pr.parsing_used,
        parsing_backend=pr.parsing_backend,
        passes_gate=True,
        issues=issues,
        landmark_backend=settings.landmark_backend,
        llm_used=llm_used,
    )

    return _attach_debug(
        AnalysisResponse(
            photo_quality=pq,
            metrics=metrics,
            confidence=conf,
            recommendations=recs,
        ),
        run_dir,
    )


def health_status() -> dict:
    from app.services.product_catalog import catalog_stats

    return {
        "status": "ok",
        "rules_version": settings.rules_version,
        "parsing": parsing_health(),
        "landmark_backend": settings.landmark_backend,
        "parsing_backend_default": settings.parsing_backend,
        "llm_available": settings.llm_available,
        "generative_available": settings.generative_available,
        "products_catalog": catalog_stats(),
    }
