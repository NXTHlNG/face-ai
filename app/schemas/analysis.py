from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PhotoQuality(BaseModel):
    sharpness_score: float = Field(..., ge=0, le=1)
    exposure_score: float = Field(..., ge=0, le=1)
    face_coverage_ratio: float = Field(..., ge=0, le=1)
    passes_gate: bool
    issues: list[str] = Field(default_factory=list)


class GeometryMetrics(BaseModel):
    cheekbone_width_norm: float
    jaw_to_cheek_ratio: float
    face_aspect_ratio: float
    forehead_to_cheek_ratio: float
    vertical_thirds_balance: float = Field(..., ge=0, le=1)
    face_shape: Literal[
        "oval", "round", "square", "heart", "triangle", "oblong", "diamond", "unknown"
    ]
    face_shape_confidence: float = Field(..., ge=0, le=1)
    geometry_landmark_set: str | None = None


class ContrastMetrics(BaseModel):
    skin_L_mean: float
    brow_L_mean: float
    hair_L_mean: float | None = None
    iris_L_mean: float | None = None
    lip_L_mean: float | None = None
    brow_skin_delta_L: float
    hair_skin_delta_L: float | None = None
    iris_skin_delta_L: float | None = None
    lip_skin_delta_L: float | None = None
    value_contrast_index: float = Field(..., ge=0, le=100)
    contrast_bucket: Literal["low", "medium", "high"]
    parsing_used: bool = False


class ColorFeatures(BaseModel):
    skin_ab_mean: tuple[float, float]
    skin_chroma_hint: float
    hair_L_mean: float | None = None
    hair_ab_mean: tuple[float, float] | None = None
    iris_ab_mean: tuple[float, float] | None = None
    eye_color_hint: Literal[
        "blue", "gray", "green", "hazel", "amber",
        "light_brown", "brown", "dark_brown", "unknown",
    ]
    eye_color_label_ru: str
    eye_color_confidence: float = Field(..., ge=0, le=1)
    undertone_hint: Literal["warm", "cool", "neutral"]
    depth_hint: Literal["light", "deep", "medium"]
    seasonal_twelve: Literal[
        "light_spring", "true_spring", "bright_spring",
        "light_summer", "true_summer", "soft_summer",
        "soft_autumn", "true_autumn", "deep_autumn",
        "deep_winter", "true_winter", "bright_winter", "unknown",
    ]
    seasonal_twelve_confidence: float = Field(..., ge=0, le=1)
    seasonal_guess: Literal["spring", "summer", "autumn", "winter", "unknown"]
    seasonal_confidence: float = Field(..., ge=0, le=1)


class SeasonScore(BaseModel):
    season: str
    probability: float = Field(..., ge=0, le=1)


class SubtypeScore(BaseModel):
    subtype: str
    probability: float = Field(..., ge=0, le=1)


class MunsellScores(BaseModel):
    undertone: int = Field(..., ge=1, le=5)
    chroma: int = Field(..., ge=1, le=5)
    value: int = Field(..., ge=1, le=5)
    contrast: int = Field(..., ge=1, le=5)


class PaletteSwatch(BaseModel):
    region: str
    rgb: tuple[int, int, int]
    hex: str
    lab_opencv: tuple[float, float, float]
    lab_cielab: tuple[float, float, float] | None = None


class AnalysisPalette(BaseModel):
    """Colors extracted from the face and season reference swatches used in classification."""
    face: list[PaletteSwatch] = Field(default_factory=list)
    season_reference: list[PaletteSwatch] | None = None


class SeasonalAnalysis(BaseModel):
    seasonal_guess: Literal["spring", "summer", "autumn", "winter", "unknown"]
    seasonal_twelve: Literal[
        "light_spring", "true_spring", "bright_spring",
        "light_summer", "true_summer", "soft_summer",
        "soft_autumn", "true_autumn", "deep_autumn",
        "deep_winter", "true_winter", "bright_winter", "unknown",
    ]
    seasonal_sixteen: str
    seasonal_confidence: float = Field(..., ge=0, le=1)
    seasonal_twelve_confidence: float = Field(..., ge=0, le=1)
    seasonal_guess_top_k: list[SeasonScore] = Field(default_factory=list)
    seasonal_twelve_top_k: list[SubtypeScore] = Field(default_factory=list)
    munsell_scores: MunsellScores | dict[str, int] | None = None
    skin_tone_class: str | None = None
    undertone_hint: Literal["warm", "cool", "neutral"]
    undertone_source: Literal["face", "wrist", "fused", "park_skin_ab"] = "face"
    classifier_contributors: dict[str, Any] = Field(default_factory=dict)
    delta_e_scores: dict[str, float] | None = None
    seasonal_method: str = "ensemble"


class MaskPreview(BaseModel):
    skin_mask_b64: str | None = None
    combined_features_b64: str | None = None


class ParsingMeta(BaseModel):
    used_model: bool
    parsing_backend: Literal[
        "bisenet_resnet34", "farl_b", "segface", "landmark_fallback"
    ] = "landmark_fallback"
    glasses_mask_ratio: float | None = None


class ConfidenceBreakdown(BaseModel):
    overall: float = Field(..., ge=0, le=1)
    geometry: float = Field(..., ge=0, le=1)
    contrast: float = Field(..., ge=0, le=1)
    color: float = Field(..., ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    category: Literal[
        "glasses", "hair", "makeup", "clothing_colors", "jewelry", "general"
    ]
    title: str
    detail: str
    rule_id: str
    rule_version: str
    based_on: dict[str, Any] = Field(default_factory=dict)


class BackendOverrides(BaseModel):
    parsing_backend: Literal[
        "bisenet_resnet34", "farl_b", "segface", "landmark_fallback"
    ] | None = None
    skin_color_backend: Literal["mean_lab", "xmeans_hsv_deltae"] | None = None
    season_classifier: Literal["munsell_lookup", "ensemble", "park_imcom18"] | None = None


class AnalysisRequestMeta(BaseModel):
    daylight_hint: bool | None = None
    no_makeup_hint: bool | None = None
    backends: BackendOverrides | None = None


class MetricsBundle(BaseModel):
    geometry: GeometryMetrics
    contrast: ContrastMetrics
    color_features: ColorFeatures
    parsing: ParsingMeta
    seasonal: SeasonalAnalysis | None = None
    analysis_palette: AnalysisPalette | None = None
    mask_preview: MaskPreview | None = None


class DebugArtifact(BaseModel):
    name: str
    kind: Literal["image", "text", "json"]


class DebugBundle(BaseModel):
    run_id: str
    artifacts: list[DebugArtifact] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    photo_quality: PhotoQuality
    metrics: MetricsBundle | None = None
    confidence: ConfidenceBreakdown
    recommendations: list[RecommendationItem]
    debug: DebugBundle | None = None
    disclaimer: str = Field(
        default="Рекомендации носят ориентировочный характер; цвет и контраст зависят от освещения и камеры.",
    )
