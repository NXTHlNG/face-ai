from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PhotoQuality(BaseModel):
    """Эвристики качества входного изображения."""

    sharpness_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Нормализованная резкость (Laplacian variance).",
    )
    exposure_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Насколько экспозиция в допустимом диапазоне.",
    )
    face_coverage_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Доля площади изображения, занятая bbox лица.",
    )
    passes_gate: bool = Field(
        ...,
        description="Проходит ли фото минимальный порог для анализа.",
    )
    issues: list[str] = Field(default_factory=list)


class GeometryMetrics(BaseModel):
    cheekbone_width_norm: float = Field(..., description="Ширина скул / межзрачковое (условно норм.).")
    jaw_to_cheek_ratio: float = Field(..., description="Отношение ширины челюсти к ширине скул.")
    face_aspect_ratio: float = Field(..., description="Высота лица / ширина (ограничивающий контур).")
    forehead_to_cheek_ratio: float = Field(..., description="Ширина лба / ширина скул.")
    vertical_thirds_balance: float = Field(
        ...,
        ge=0,
        le=1,
        description="Насколько равны вертикальные трети (1 — идеально ровно).",
    )
    face_shape: Literal[
        "oval",
        "round",
        "square",
        "heart",
        "triangle",
        "oblong",
        "diamond",
        "unknown",
    ]
    face_shape_confidence: float = Field(..., ge=0, le=1)
    geometry_landmark_set: str | None = Field(
        None,
        description="Источник точек: mediapipe_478 или dlib_iBUG_81.",
    )


class ContrastMetrics(BaseModel):
    """Контраст между элементами лица (не HDR)."""

    skin_L_mean: float
    brow_L_mean: float
    hair_L_mean: float | None = None
    iris_L_mean: float | None = Field(None, description="Средняя светлота L* радужки (оценка по landmarks / парсингу).")
    lip_L_mean: float | None = Field(None, description="Средняя светлота L* губ (парсинг или контур рта).")
    brow_skin_delta_L: float = Field(..., description="|L брови − L кожи|.")
    hair_skin_delta_L: float | None = Field(None, description="|L волос − L кожи|.")
    iris_skin_delta_L: float | None = Field(None, description="|L радужки − L кожи|.")
    lip_skin_delta_L: float | None = Field(None, description="|L губ − L кожи|.")
    value_contrast_index: float = Field(
        ...,
        ge=0,
        le=100,
        description="Сводный индекс value-contrast (волосы, брови, глаза, губы vs кожа), 0–100.",
    )
    contrast_bucket: Literal["low", "medium", "high"]
    parsing_used: bool = Field(False, description="Использовалась ли сегментационная модель.")


class ColorFeatures(BaseModel):
    """Признаки для палитры / грубого цветотипа (ориентиры, не диагноз)."""

    skin_ab_mean: tuple[float, float] = Field(..., description="Средние a*, b* по коже (Lab).")
    skin_chroma_hint: float = Field(..., description="Грубая насыщенность кожи.")
    hair_L_mean: float | None = None
    hair_ab_mean: tuple[float, float] | None = None
    iris_ab_mean: tuple[float, float] | None = None
    eye_color_hint: Literal[
        "blue",
        "gray",
        "green",
        "hazel",
        "amber",
        "light_brown",
        "brown",
        "dark_brown",
        "unknown",
    ] = Field(
        ...,
        description="Грубая категория цвета глаз по LAB радужки (освещение и очки искажают).",
    )
    eye_color_label_ru: str = Field(
        ...,
        description="Человекочитаемая подпись цвета глаз на русском.",
    )
    eye_color_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Уверенность в eye_color_hint (выше при явном снимке радужки без очков).",
    )
    undertone_hint: Literal["warm", "cool", "neutral"] = Field(
        ...,
        description="Подтон: кожа + при наличии волосы и радужка в a*b* (освещение искажает).",
    )
    depth_hint: Literal["light", "deep", "medium"] = Field(
        ...,
        description="Глубина (value): светлота кожи и волос в LAB.",
    )
    seasonal_twelve: Literal[
        "light_spring",
        "true_spring",
        "bright_spring",
        "light_summer",
        "true_summer",
        "soft_summer",
        "soft_autumn",
        "true_autumn",
        "deep_autumn",
        "deep_winter",
        "true_winter",
        "bright_winter",
        "unknown",
    ] = Field(
        ...,
        description="12 сезонов (англ. ключи): подтип внутри тёплого/холодного × светлота × soft/clear.",
    )
    seasonal_twelve_confidence: float = Field(..., ge=0, le=1)
    seasonal_guess: Literal[
        "spring",
        "summer",
        "autumn",
        "winter",
        "unknown",
    ] = Field(
        ...,
        description="Родительский сезон (4), согласован с seasonal_twelve.",
    )
    seasonal_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Уверенность в подтипе (seasonal_twelve); дублируется в seasonal_confidence для совместимости.",
    )


class ConfidenceBreakdown(BaseModel):
    overall: float = Field(..., ge=0, le=1)
    geometry: float = Field(..., ge=0, le=1)
    contrast: float = Field(..., ge=0, le=1)
    color: float = Field(..., ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    category: Literal[
        "glasses",
        "hair",
        "makeup",
        "clothing_colors",
        "jewelry",
        "general",
    ]
    title: str
    detail: str
    rule_id: str = Field(..., description="Стабильный идентификатор правила.")
    rule_version: str = Field(..., description="Версия набора правил.")
    based_on: dict[str, Any] = Field(
        default_factory=dict,
        description="Какие поля метрик задействованы (для объяснимости).",
    )


class AnalysisRequestMeta(BaseModel):
    """Опциональные подсказки клиента (калибровка в будущем)."""

    daylight_hint: bool | None = None
    no_makeup_hint: bool | None = None


class ParsingMeta(BaseModel):
    used_model: bool
    glasses_mask_ratio: float | None = None


class MetricsBundle(BaseModel):
    geometry: GeometryMetrics
    contrast: ContrastMetrics
    color_features: ColorFeatures
    parsing: ParsingMeta


class AnalysisResponse(BaseModel):
    photo_quality: PhotoQuality
    metrics: MetricsBundle | None = Field(
        None,
        description="Типизированные метрики; null если лицо не найдено или сработал gate качества.",
    )
    confidence: ConfidenceBreakdown
    recommendations: list[RecommendationItem]
    disclaimer: str = Field(
        default="Рекомендации носят ориентировочный характер; цвет и контраст зависят от освещения и камеры.",
    )
