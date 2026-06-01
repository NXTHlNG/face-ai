from __future__ import annotations

import numpy as np

from app.backends.parsing.types import ParsingResult
from app.backends.try_on.generative_api import GenerativeModelAPI, resolve_mask_policy, save_try_on_debug
from app.backends.try_on import mask_builder
from app.backends.try_on.prompt_builder import build_prompt
from app.config import settings
from app.schemas.try_on import TryOnCategory, TryOnCategoryMeta
from app.services.debug_dump import new_run_dir


def render_category_generative(
    api: GenerativeModelAPI,
    image_bgr: np.ndarray,
    pr: ParsingResult,
    landmarks_px: np.ndarray | None,
    category: TryOnCategory,
    *,
    season_twelve: str,
    season_guess: str = "",
    product_skus: dict[str, str] | None = None,
    use_mask: bool = True,
) -> tuple[np.ndarray | None, TryOnCategoryMeta | None]:
    shape = image_bgr.shape
    zones: list[str] = []
    sku: str | None = None

    if category == "makeup":
        mask, zones = mask_builder.build_makeup_mask(pr, shape, landmarks_px)
        if product_skus:
            sku = product_skus.get("makeup") or product_skus.get("lipstick")
    elif category == "hairstyle":
        mask = mask_builder.build_hairstyle_mask(pr, shape)
        if product_skus:
            sku = product_skus.get("hairstyle")
    elif category == "glasses":
        mask = mask_builder.build_glasses_mask(pr, shape, landmarks_px)
        if product_skus:
            sku = product_skus.get("glasses") or product_skus.get("frames")
    else:
        return None, None

    composite_requested = category == "makeup" and settings.generative_composite_mask
    api_use_mask, composite_locally = resolve_mask_policy(
        transport=api.transport,
        model=api.model,
        use_mask=use_mask,
        composite_requested=composite_requested,
    )
    needs_mask = api_use_mask or composite_locally
    if needs_mask and int(np.max(mask)) < 1:
        return None, None

    prompt, negative = build_prompt(
        category,
        season_twelve=season_twelve,
        season_guess=season_guess,
        product_skus=product_skus,
        use_mask=api_use_mask,
    )
    debug_ctx: dict = {}
    run_dir = new_run_dir() if settings.debug_save_images else None
    result = api.render(
        image_bgr,
        mask if api_use_mask else None,
        prompt,
        negative_prompt=negative,
        use_mask=api_use_mask,
        composite_mask=mask if composite_locally else None,
        debug_ctx=debug_ctx,
    )
    if result is None:
        return None, None

    if run_dir is not None:
        save_try_on_debug(
            run_dir,
            category=category,
            image_bgr=image_bgr,
            mask=mask,
            zones=zones,
            model_raw_bgr=debug_ctx.get("model_raw_bgr"),
            composite_applied=bool(debug_ctx.get("composite_applied")),
            api_use_mask=bool(debug_ctx.get("api_use_mask")),
        )

    meta = TryOnCategoryMeta(
        renderer="generative",
        zones=zones,
        sku=sku,
        type=category,
        masked=api_use_mask or composite_locally,
    )
    return result, meta
