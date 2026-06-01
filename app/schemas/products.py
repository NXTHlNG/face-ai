from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProductCategory = Literal[
    "lipstick",
    "foundation",
    "blush",
    "frames",
    "hairstyle",
]


class ProductSku(BaseModel):
    sku: str
    brand: str
    name: str
    category: ProductCategory
    lab: tuple[float, float, float] = Field(..., description="CIELAB L*, a*, b*")
    season_tags: list[str] = Field(default_factory=list)
    overlay_asset: str | None = None
    rgb: tuple[int, int, int] | None = None


class ProductCatalog(BaseModel):
    version: str = "1.0"
    products: list[ProductSku]


class ProductMatchItem(BaseModel):
    sku: str
    brand: str
    name: str
    category: ProductCategory
    delta_e: float = Field(..., ge=0)
    match_pct: float = Field(..., ge=0, le=100)
    overlay_asset: str | None = None
    rgb: tuple[int, int, int] | None = None


class ProductMatchResponse(BaseModel):
    season_twelve: str
    category: ProductCategory
    target_lab: tuple[float, float, float] | None = None
    season_expanded: bool = False
    matches: list[ProductMatchItem]
