import base64

import cv2
import numpy as np
import pytest

from app.config import settings
from app.backends.try_on.generative_api import (
    GenerativeModelAPI,
    composite_masked,
    parse_custom_json_response,
    parse_openai_images_response,
    resolve_generative_url,
    resolve_mask_policy,
    resolve_openai_response_format,
    unwrap_diptych_if_present,
)


def _sample_bgr() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 20:44] = (180, 120, 90)
    return img


def _sample_mask() -> np.ndarray:
    m = np.zeros((64, 64), dtype=np.uint8)
    m[24:40, 24:40] = 255
    return m


def test_resolve_generative_url_openai_base():
    assert resolve_generative_url("https://api.openai.com/v1", "openai_images_edit").endswith(
        "/images/edits"
    )


def test_resolve_generative_url_custom():
    url = "http://localhost:8090/inpaint"
    assert resolve_generative_url(url, "custom_json") == url


def test_parse_openai_images_response():
    body = {"data": [{"b64_json": "abc"}]}
    assert parse_openai_images_response(body) == "abc"


def test_parse_custom_json_response():
    body = {"image_b64": "xyz"}
    assert parse_custom_json_response(body) == "xyz"


def test_render_openai_success():
    img = _sample_bgr()
    mask = _sample_mask()
    ok, buf = cv2.imencode(".jpg", img)
    out_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    def fake_multipart(url, fields, files, headers, timeout_s):
        return {"data": [{"b64_json": out_b64}]}

    api = GenerativeModelAPI(
        api_url="https://example.com/v1",
        api_key="k",
        transport="openai_images_edit",
        post_multipart=fake_multipart,
    )
    result = api.render(img, mask, "test prompt")
    assert result is not None
    assert result.shape == img.shape


def test_render_returns_none_on_failure():
    def fail(*_a, **_k):
        raise RuntimeError("boom")

    api = GenerativeModelAPI(
        api_url="https://example.com/v1",
        transport="openai_images_edit",
        post_multipart=fail,
    )
    assert api.render(_sample_bgr(), _sample_mask(), "p") is None


def test_api_key_falls_back_to_llm_key(monkeypatch):
    monkeypatch.setattr(settings, "generative_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "sk-from-llm")
    api = GenerativeModelAPI(api_url="https://example.com/v1")
    assert api.api_key == "sk-from-llm"


def test_render_skips_without_any_key(monkeypatch):
    monkeypatch.setattr(settings, "generative_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    api = GenerativeModelAPI(api_url="https://example.com/v1")
    assert api.render(_sample_bgr(), _sample_mask(), "p") is None


def test_align_to_source_size():
    from app.backends.try_on.generative_api import _align_to_source_size

    src = np.zeros((100, 200, 3), dtype=np.uint8)
    out = np.zeros((512, 512, 3), dtype=np.uint8)
    aligned = _align_to_source_size(out, src.shape)
    assert aligned.shape == (100, 200, 3)


def test_resolve_openai_response_format_auto():
    assert resolve_openai_response_format("dall-e-2", "auto") == "b64_json"
    assert resolve_openai_response_format("gpt-image-2", "auto") is None
    assert resolve_openai_response_format("gpt-image-2", "omit") is None
    assert resolve_openai_response_format("gpt-image-2", "b64_json") == "b64_json"


def test_parse_openai_images_response_url(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url

        class Resp:
            def read(self):
                ok, buf = cv2.imencode(".jpg", _sample_bgr())
                return buf.tobytes()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return Resp()

    monkeypatch.setattr(
        "app.backends.try_on.generative_api.request.urlopen",
        fake_urlopen,
    )
    body = {"data": [{"url": "https://cdn.example/out.png"}]}
    b64 = parse_openai_images_response(body)
    assert captured["url"] == "https://cdn.example/out.png"
    assert _decode_roundtrip(b64) is not None


def _decode_roundtrip(b64: str) -> np.ndarray | None:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def test_render_openai_omits_response_format_for_gpt_image(monkeypatch):
    monkeypatch.setattr(settings, "generative_response_format", "auto")
    img = _sample_bgr()
    captured: dict = {}

    def fake_multipart(url, fields, files, headers, timeout_s):
        captured["fields"] = dict(fields)
        ok, buf = cv2.imencode(".jpg", img)
        return {"data": [{"b64_json": base64.b64encode(buf.tobytes()).decode("ascii")}]}

    api = GenerativeModelAPI(
        api_url="https://example.com/v1",
        model="gpt-image-2",
        post_multipart=fake_multipart,
    )
    api.render(img, _sample_mask(), "p")
    assert "response_format" not in captured["fields"]


def test_render_openai_sends_size_auto(monkeypatch):
    monkeypatch.setattr(settings, "generative_image_size", "auto")
    img = _sample_bgr()
    captured: dict = {}

    def fake_multipart(url, fields, files, headers, timeout_s):
        captured["fields"] = dict(fields)
        ok, buf = cv2.imencode(".jpg", img)
        return {"data": [{"b64_json": base64.b64encode(buf.tobytes()).decode("ascii")}]}

    api = GenerativeModelAPI(api_url="https://example.com/v1", post_multipart=fake_multipart)
    api.render(img, _sample_mask(), "p")
    assert captured["fields"].get("size") == "auto"


def test_render_openai_without_mask():
    img = _sample_bgr()
    captured: dict = {}

    def fake_multipart(url, fields, files, headers, timeout_s):
        captured["files"] = [name for name, *_ in files]
        ok, buf = cv2.imencode(".jpg", img)
        return {"data": [{"b64_json": base64.b64encode(buf.tobytes()).decode("ascii")}]}

    api = GenerativeModelAPI(
        api_url="https://example.com/v1",
        post_multipart=fake_multipart,
    )
    result = api.render(img, None, "edit face", use_mask=False)
    assert result is not None
    assert captured["files"] == ["image"]


def test_composite_masked_preserves_outside_region():
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    original[:, :] = (100, 100, 100)
    edited = original.copy()
    edited[20:44, 20:44] = (0, 0, 255)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    out = composite_masked(original, edited, mask)
    assert tuple(out[0, 0]) == (100, 100, 100)
    assert not np.array_equal(out[32, 32], original[32, 32])


def test_render_applies_composite_mask(monkeypatch):
    monkeypatch.setattr(settings, "generative_composite_mask", True)
    img = _sample_bgr()
    mask = _sample_mask()
    edited = img.copy()
    edited[:, :] = (0, 0, 255)

    def fake_multipart(url, fields, files, headers, timeout_s):
        ok, buf = cv2.imencode(".jpg", edited)
        return {"data": [{"b64_json": base64.b64encode(buf.tobytes()).decode("ascii")}]}

    api = GenerativeModelAPI(
        api_url="https://example.com/v1",
        model="dall-e-2",
        post_multipart=fake_multipart,
    )
    result = api.render(img, None, "p", use_mask=False, composite_mask=mask)
    assert result is not None
    assert tuple(result[0, 0]) == tuple(img[0, 0])
    assert not np.array_equal(result[32, 32], img[32, 32])


def test_resolve_mask_policy_gemini_openai_skips_mask_and_composite():
    api_mask, composite = resolve_mask_policy(
        transport="openai_images_edit",
        model="google/gemini-3-pro-image-preview",
        use_mask=True,
        composite_requested=True,
    )
    assert api_mask is False
    assert composite is False


def test_resolve_mask_policy_custom_json_honors_flags():
    api_mask, composite = resolve_mask_policy(
        transport="custom_json",
        model="google/gemini-3-pro-image-preview",
        use_mask=True,
        composite_requested=True,
    )
    assert api_mask is True
    assert composite is True


def test_unwrap_diptych_horizontal():
    src_shape = (400, 300, 3)
    panel = np.zeros((400, 300, 3), dtype=np.uint8)
    panel[:, :] = (0, 0, 200)
    diptych = np.hstack([np.zeros_like(panel), panel])
    out, was = unwrap_diptych_if_present(diptych, src_shape)
    assert was is True
    assert out.shape[1] == 300
    assert tuple(out[0, 0]) == (200, 0, 0)


def test_unwrap_diptych_skips_normal_output():
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    out, was = unwrap_diptych_if_present(img, img.shape)
    assert was is False
    assert out is img


def test_render_custom_json():
    img = _sample_bgr()
    mask = _sample_mask()
    ok, buf = cv2.imencode(".jpg", img)
    out_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    def fake_json(url, payload, headers, timeout_s):
        assert "image_b64" in payload
        assert "mask_b64" in payload
        return {"image_b64": out_b64}

    api = GenerativeModelAPI(
        api_url="http://127.0.0.1:8090/inpaint",
        transport="custom_json",
        post_json=fake_json,
    )
    result = api.render(img, mask, "prompt", negative_prompt="neg")
    assert result is not None
