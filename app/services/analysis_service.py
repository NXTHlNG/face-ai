from __future__ import annotations

from app.pipeline.intake import decode_image_bytes
from app.pipeline.orchestrator import AnalyzeOptions, analyze_bgr
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisRequestMeta,
    BackendOverrides,
    ConfidenceBreakdown,
    PhotoQuality,
)
def analyze_image_bytes(
    data: bytes,
    *,
    wrist_data: bytes | None = None,
    meta: AnalysisRequestMeta | None = None,
) -> AnalysisResponse:
    image_bgr = decode_image_bytes(data)
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
                overall=0.0, geometry=0.0, contrast=0.0, color=0.0, notes=["invalid_image"]
            ),
            recommendations=[],
        )

    wrist_bgr = None
    if wrist_data:
        wrist_bgr = decode_image_bytes(wrist_data)

    opts = AnalyzeOptions(wrist_bgr=wrist_bgr, backends=meta.backends if meta else None)
    if meta and meta.backends and meta.backends.parsing_backend:
        opts.parsing_backend = meta.backends.parsing_backend

    return analyze_bgr(image_bgr, opts)
