from __future__ import annotations

import json
from functools import lru_cache

from app.config import settings
from app.schemas.products import ProductCatalog, ProductSku


class ProductCatalogError(Exception):
    pass


@lru_cache(maxsize=1)
def load_catalog() -> ProductCatalog:
    path = settings.resolved_products_path()
    if not path.is_file():
        raise ProductCatalogError(f"products catalog not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ProductCatalog.model_validate(raw)


def list_products(*, category: str | None = None) -> list[ProductSku]:
    catalog = load_catalog()
    if category is None:
        return list(catalog.products)
    return [p for p in catalog.products if p.category == category]


def get_product(sku: str) -> ProductSku | None:
    for product in load_catalog().products:
        if product.sku == sku:
            return product
    return None


def catalog_stats() -> dict[str, int | str]:
    try:
        catalog = load_catalog()
    except ProductCatalogError:
        return {"loaded": False, "count": 0, "version": ""}
    by_cat: dict[str, int] = {}
    for p in catalog.products:
        by_cat[p.category] = by_cat.get(p.category, 0) + 1
    return {
        "loaded": True,
        "count": len(catalog.products),
        "version": catalog.version,
        "by_category": by_cat,
    }


def reload_catalog() -> None:
    load_catalog.cache_clear()
