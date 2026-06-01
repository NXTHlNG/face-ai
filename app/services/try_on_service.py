from __future__ import annotations

import cv2
import numpy as np

from app.backends.try_on.engine import TryOnEngine
from app.backends.try_on import mask_builder
from app.config import settings
from app.pipeline.face_prepare import prepare_face_bgr
from app.schemas.try_on import TryOnCategory, TryOnPhotoRequestMeta, TryOnPhotoResult


class TryOnError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _decode_image_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise TryOnError("invalid image file", status_code=400)
    return image


def _default_categories() -> list[TryOnCategory]:
    raw = settings.tryon_default_categories.strip()
    if not raw:
        return ["makeup", "glasses", "hairstyle"]
    out: list[TryOnCategory] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if key in ("makeup", "glasses", "hairstyle"):
            out.append(key)  # type: ignore[arg-type]
    return out or ["makeup", "glasses", "hairstyle"]


def _mask_nonempty_for_category(
    category: TryOnCategory,
    pr,
    shape: tuple[int, ...],
    landmarks_px: np.ndarray | None,
) -> bool:
    if category == "makeup":
        mask, _ = mask_builder.build_makeup_mask(pr, shape, landmarks_px)
    elif category == "hairstyle":
        mask = mask_builder.build_hairstyle_mask(pr, shape)
    else:
        mask = mask_builder.build_glasses_mask(pr, shape, landmarks_px)
    return int(np.max(mask)) > 0


def _resolve_use_mask(meta: TryOnPhotoRequestMeta) -> bool:
    if meta.use_mask is not None:
        return meta.use_mask
    return settings.generative_use_mask


def try_on_photo_bytes(data: bytes, meta: TryOnPhotoRequestMeta) -> TryOnPhotoResult:
    image_bgr = _decode_image_bgr(data)
    prepared = prepare_face_bgr(image_bgr, skip_quality_gate=True)
    if prepared is None:
        raise TryOnError("no face detected", status_code=400)

    categories = meta.categories or _default_categories()
    if not categories:
        raise TryOnError("categories list is empty", status_code=400)

    use_mask = _resolve_use_mask(meta)
    if use_mask:
        usable = [
            c
            for c in categories
            if _mask_nonempty_for_category(
                c, prepared.parsing, image_bgr.shape, prepared.landmarks_px
            )
        ]
        if not usable:
            raise TryOnError(
                "parsing masks insufficient for requested try-on categories",
                status_code=400,
            )
    else:
        usable = list(categories)

    engine = TryOnEngine()
    result = engine.render_photo(
        image_bgr,
        prepared.parsing,
        prepared.landmarks_px,
        meta.season_twelve,
        categories=usable,
        product_skus=meta.product_skus,
        use_generative=meta.generative,
        use_mask=use_mask,
    )
    if result.generative is None and meta.generative:
        if not settings.generative_available:
            raise TryOnError(
                "generative API not configured (FACE_AI_GENERATIVE_API_URL=none)",
                status_code=503,
            )
        raise TryOnError("generative rendering failed for all categories", status_code=502)
    return result
