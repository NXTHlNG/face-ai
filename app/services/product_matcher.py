from __future__ import annotations

import json
from functools import lru_cache

from app.backends.color.delta_e import delta_e_cie2000, rgb_to_lab
from app.backends.season.season_maps import TWELVE_TO_FOUR
from app.config import settings
from app.schemas.products import ProductMatchItem, ProductMatchResponse, ProductSku
from app.services.product_catalog import list_products


def delta_e_to_match_pct(delta_e: float) -> float:
    """Map CIEDE2000 distance to 0–100 match score (heuristic for demo UI)."""
    return round(max(0.0, min(100.0, 100.0 - delta_e * 2.5)), 1)


def _default_target_lab(season_twelve: str) -> tuple[float, float, float]:
    path = settings.data_dir / "reference_swatches.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        parent = TWELVE_TO_FOUR.get(season_twelve, "unknown")
        anchors = raw.get("season_rgb_anchors", {}).get(parent)
        if anchors:
            rgb = anchors[0]
            return rgb_to_lab(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return (50.0, 0.0, 0.0)


@lru_cache(maxsize=1)
def load_makeup_db() -> dict:
    path = settings.resolved_makeup_db_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def season_zone_lab(season_twelve: str, zone: str) -> tuple[float, float, float] | None:
    db = load_makeup_db()
    entry = db.get(season_twelve, {})
    lab = entry.get(zone)
    if lab is None or len(lab) != 3:
        return None
    return (float(lab[0]), float(lab[1]), float(lab[2]))


def _season_priority(
    product: ProductSku,
    season_twelve: str,
    parent: str | None,
) -> int:
    if season_twelve in product.season_tags:
        return 0
    if parent and parent in product.season_tags:
        return 1
    return 2


def match_products(
    season_twelve: str,
    category: str,
    *,
    target_lab: tuple[float, float, float] | None = None,
    top_k: int = 3,
) -> ProductMatchResponse:
    if top_k < 1:
        top_k = 1
    pool = list_products(category=category)
    parent = TWELVE_TO_FOUR.get(season_twelve)
    ref_lab = target_lab if target_lab is not None else _default_target_lab(season_twelve)

    ranked_pairs: list[tuple[ProductSku, float, int]] = []
    for product in pool:
        de = delta_e_cie2000(ref_lab, product.lab)
        pri = _season_priority(product, season_twelve, parent)
        ranked_pairs.append((product, de, pri))
    ranked_pairs.sort(key=lambda x: (x[2], x[1]))

    top = ranked_pairs[:top_k]
    season_expanded = bool(top) and top[0][2] > 0

    matches: list[ProductMatchItem] = []
    for product, de, _pri in top:
        matches.append(
            ProductMatchItem(
                sku=product.sku,
                brand=product.brand,
                name=product.name,
                category=product.category,
                delta_e=round(de, 2),
                match_pct=delta_e_to_match_pct(de),
                overlay_asset=product.overlay_asset,
                rgb=product.rgb,
            )
        )
    return ProductMatchResponse(
        season_twelve=season_twelve,
        category=category,  # type: ignore[arg-type]
        target_lab=ref_lab,
        season_expanded=season_expanded,
        matches=matches,
    )
