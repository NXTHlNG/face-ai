from __future__ import annotations

import base64

import cv2
import numpy as np

from app.backends.parsing.types import ParsingResult
from app.backends.try_on.generative_api import GenerativeModelAPI
from app.backends.try_on.renderers.category import render_category_generative
from app.schemas.try_on import (
    TryOnActiveMode,
    TryOnBranchResult,
    TryOnCategory,
    TryOnCategoryMeta,
    TryOnPhotoResult,
)

_COMPOSITE_ORDER: tuple[TryOnCategory, ...] = ("hairstyle", "makeup", "glasses")


def _encode_jpeg_b64(image_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class TryOnEngine:
    def __init__(self, generative_api: GenerativeModelAPI | None = None) -> None:
        self._gen = generative_api or GenerativeModelAPI()

    def render_photo(
        self,
        image_bgr: np.ndarray,
        parsing: ParsingResult,
        landmarks_px: np.ndarray | None,
        season_twelve: str,
        *,
        categories: list[TryOnCategory],
        product_skus: dict[str, str] | None = None,
        season_guess: str = "",
        use_generative: bool = True,
        use_mask: bool = True,
    ) -> TryOnPhotoResult:
        original_b64 = _encode_jpeg_b64(image_bgr)
        generative_branch: TryOnBranchResult | None = None
        active_mode: TryOnActiveMode = "cv"

        if use_generative and self._gen.available:
            work = image_bgr.copy()
            cat_meta: dict[str, TryOnCategoryMeta] = {}
            for category in _COMPOSITE_ORDER:
                if category not in categories:
                    continue
                out, meta = render_category_generative(
                    self._gen,
                    work,
                    parsing,
                    landmarks_px,
                    category,
                    season_twelve=season_twelve,
                    season_guess=season_guess,
                    product_skus=product_skus,
                    use_mask=use_mask,
                )
                if out is not None and meta is not None:
                    work = out
                    cat_meta[category] = meta
            if cat_meta:
                generative_branch = TryOnBranchResult(
                    composite_b64=_encode_jpeg_b64(work),
                    categories=cat_meta,
                )
                active_mode = "generative"

        return TryOnPhotoResult(
            original_b64=original_b64,
            cv=None,
            generative=generative_branch,
            active_mode=active_mode,
        )
