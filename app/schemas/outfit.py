from __future__ import annotations

from pydantic import BaseModel, Field


class OutfitColorCluster(BaseModel):
    hex: str
    lab: tuple[float, float, float]
    pixel_ratio: float = Field(..., ge=0, le=1)
    season_delta_e: float = Field(..., ge=0)


class OutfitProductAlternative(BaseModel):
    sku: str
    name: str
    match_pct: float = Field(..., ge=0, le=100)


class OutfitHint(BaseModel):
    visible: bool = False
    compatibility_score: float | None = Field(default=None, ge=0, le=100)
    message: str | None = None


class OutfitScanResult(BaseModel):
    compatibility_score: float = Field(..., ge=0, le=100)
    dominant_colors: list[OutfitColorCluster] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    product_alternatives: list[OutfitProductAlternative] = Field(default_factory=list)
