from unittest.mock import MagicMock

import numpy as np

from app.backends.parsing.types import ParsingResult
from app.backends.try_on.engine import TryOnEngine
from app.backends.try_on.generative_api import GenerativeModelAPI


def _parsing(h: int = 64, w: int = 64) -> ParsingResult:
    def zone(y0, y1, x0, x1):
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 255
        return m

    return ParsingResult(
        skin_mask=zone(20, 60, 15, 50),
        hair_mask=zone(5, 25, 15, 50),
        brow_mask=zone(28, 34, 20, 45),
        eye_glass_mask=None,
        lip_mask=zone(45, 52, 25, 40),
        eye_region_mask=zone(30, 40, 18, 47),
        parsing_used=True,
        label_map=None,
    )


def test_engine_composite_order_calls_render_per_category():
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    calls: list[str] = []

    def fake_render(image_bgr, edit_mask, prompt, *, negative_prompt="", use_mask=True):
        calls.append((prompt[:20], use_mask))
        out = image_bgr.copy()
        out[edit_mask > 127] = (255, 0, 0)
        return out

    api = GenerativeModelAPI(api_url="http://test/v1")
    api.render = fake_render  # type: ignore[method-assign]

    engine = TryOnEngine(generative_api=api)
    result = engine.render_photo(
        img,
        _parsing(),
        None,
        "light_summer",
        categories=["hairstyle", "makeup", "glasses"],
        use_generative=True,
    )
    assert result.generative is not None
    assert result.active_mode == "generative"
    assert set(result.generative.categories.keys()) == {"hairstyle", "makeup", "glasses"}
    assert len(calls) == 3


def test_engine_skips_unavailable_generative():
    api = MagicMock(spec=GenerativeModelAPI)
    api.available = False
    engine = TryOnEngine(generative_api=api)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    result = engine.render_photo(
        img,
        _parsing(),
        None,
        "true_winter",
        categories=["makeup"],
        use_generative=True,
    )
    assert result.generative is None
    assert result.active_mode == "cv"
