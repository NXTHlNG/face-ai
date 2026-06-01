from __future__ import annotations

import json
from functools import lru_cache
from app.config import settings
from app.schemas.try_on import TryOnCategory
from app.services.product_catalog import get_product

_PROMPTS_DIR = settings.data_dir / "try_on_prompts"


@lru_cache(maxsize=1)
def _load_makeup_db() -> dict[str, dict[str, list[float]]]:
    path = settings.resolved_makeup_db_path()
    return json.loads(path.read_text(encoding="utf-8"))


def _read_template(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def _lab_to_hex(lab: list[float]) -> str:
    import cv2
    import numpy as np

    lab_arr = np.array([[lab]], dtype=np.float32)
    bgr = cv2.cvtColor(lab_arr, cv2.COLOR_Lab2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return f"#{r:02x}{g:02x}{b:02x}"


def _season_palette_hint(season_twelve: str) -> str:
    db = _load_makeup_db()
    zones = db.get(season_twelve) or db.get("light_summer", {})
    parts = []
    for zone in ("lips", "blush", "shadow", "brows"):
        lab = zones.get(zone)
        if lab and len(lab) >= 3:
            parts.append(f"{zone} {_lab_to_hex(lab)} L*a*b*({lab[0]:.0f},{lab[1]:.0f},{lab[2]:.0f})")
    return "; ".join(parts)


_NO_MASK_REGION_HINT = {
    "makeup": (
        "Apply cosmetic makeup only on lips, cheeks, eyelids, and brows. "
        "Keep background, hair, and clothing unchanged."
    ),
    "hairstyle": "Change only the hair. Keep face, skin, and background identical.",
    "glasses": "Add realistic eyeglasses on the face only. Do not alter skin or background.",
}

_PRESERVATION_NEGATIVE = (
    "different person, face swap, new face, changed identity, altered head pose, "
    "head rotation, tilted head, zoom, crop, reframing, aspect ratio change, "
    "letterbox, pillarbox, stretched face, warped proportions, background change, "
    "lighting change, age change, gender change, duplicate face, extra people, "
    "before and after, before/after, side by side, side-by-side, split screen, "
    "comparison image, diptych, two panels, collage, tutorial layout, new outfit"
)


@lru_cache(maxsize=1)
def _preservation_block() -> str:
    path = _PROMPTS_DIR / "generative_preservation.txt"
    return path.read_text(encoding="utf-8").strip()


def _finalize(prompt: str, negative: str) -> tuple[str, str]:
    full_prompt = f"{prompt.strip()}\n\n{_preservation_block()}"
    full_negative = f"{negative.strip()}, {_PRESERVATION_NEGATIVE}"
    return full_prompt, full_negative


def build_prompt(
    category: TryOnCategory,
    *,
    season_twelve: str,
    season_guess: str = "",
    product_skus: dict[str, str] | None = None,
    use_mask: bool = True,
) -> tuple[str, str]:
    product_skus = product_skus or {}
    guess = season_guess or season_twelve.split("_")[0] if season_twelve else "unknown"
    palette = _season_palette_hint(season_twelve)

    if category == "makeup":
        template = _read_template("generative_makeup.txt")
        lip_sku = product_skus.get("makeup") or product_skus.get("lipstick")
        lip_hint = ""
        if lip_sku:
            p = get_product(lip_sku)
            if p:
                lip_hint = f" Lip product reference: {p.brand} {p.name}."
        prompt = template.format(
            season_twelve=season_twelve,
            season_guess=guess,
        )
        if palette:
            prompt += (
                f"\nUse only these target colors (subtle, natural finish): {palette}.{lip_hint}"
            )
        negative = (
            "plastic skin, wrong undertone, blurred eyes, changed face shape, "
            "heavy filter, unnatural makeup, sheet mask, peel-off mask, black face mask, "
            "different person, face swap, changed pose, head rotation, reframing, "
            "background change, lighting change, new portrait, re-generated photo"
        )
        if not use_mask:
            prompt += f"\n{_NO_MASK_REGION_HINT['makeup']}"
        return _finalize(prompt, negative)

    if category == "hairstyle":
        template = _read_template("generative_hairstyle.txt")
        sku = product_skus.get("hairstyle")
        name = "natural hairstyle for season"
        if sku:
            p = get_product(sku)
            if p:
                name = f"{p.name}"
        prompt = template.format(
            season_twelve=season_twelve,
            season_guess=guess,
            hairstyle_sku_name=name,
        )
        if palette:
            prompt += f"\nHair color hint from season: {palette}."
        negative = "halo around head, face distortion, background bleed, wig-like edges"
        if not use_mask:
            prompt += f"\n{_NO_MASK_REGION_HINT['hairstyle']}"
        return _finalize(prompt, negative)

    template = _read_template("generative_glasses.txt")
    sku = product_skus.get("glasses") or product_skus.get("frames")
    name = "stylish eyeglasses matching the season"
    if sku:
        p = get_product(sku)
        if p:
            name = f"{p.brand} {p.name}"
    prompt = template.format(
        season_twelve=season_twelve,
        season_guess=guess,
        glasses_sku_name=name,
    )
    negative = (
        "floating glasses, duplicate frames, distorted eyes, cartoon effect, "
        "wrong scale, missing temples"
    )

    if not use_mask:
        prompt += f"\n{_NO_MASK_REGION_HINT[category]}"
    return _finalize(prompt, negative)
