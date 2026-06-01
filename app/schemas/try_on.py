from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TryOnCategory = Literal["makeup", "glasses", "hairstyle"]
TryOnActiveMode = Literal["cv", "generative", "hybrid"]


class TryOnCategoryMeta(BaseModel):
    renderer: Literal["cv", "generative"] = "cv"
    zones: list[str] = Field(default_factory=list)
    sku: str | None = None
    type: str | None = None
    masked: bool = True


class TryOnBranchResult(BaseModel):
    composite_b64: str
    categories: dict[str, TryOnCategoryMeta] = Field(default_factory=dict)


class TryOnPhotoResult(BaseModel):
    original_b64: str
    cv: TryOnBranchResult | None = None
    generative: TryOnBranchResult | None = None
    active_mode: TryOnActiveMode = "generative"


class TryOnPhotoRequestMeta(BaseModel):
    season_twelve: str
    categories: list[TryOnCategory] = Field(
        default_factory=lambda: ["makeup", "glasses", "hairstyle"]
    )
    product_skus: dict[str, str] | None = None
    generative: bool = True
    use_mask: bool | None = None
    analyze_run_id: str | None = None
