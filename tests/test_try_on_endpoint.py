from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backends.parsing.types import ParsingResult
from app.main import app
from app.pipeline.face_prepare import PreparedFace
from app.schemas.try_on import TryOnBranchResult, TryOnCategoryMeta, TryOnPhotoResult


def _prepared() -> PreparedFace:
    h, w = 64, 64
    lip = np.zeros((h, w), dtype=np.uint8)
    lip[40:48, 24:40] = 255
    pr = ParsingResult(
        skin_mask=np.ones((h, w), dtype=np.uint8) * 255,
        hair_mask=np.zeros((h, w), dtype=np.uint8),
        brow_mask=lip.copy(),
        eye_glass_mask=None,
        lip_mask=lip,
        eye_region_mask=lip.copy(),
        parsing_used=True,
        label_map=None,
    )
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    return PreparedFace(
        image_bgr=img,
        image_rgb=img[:, :, ::-1],
        landmarks_px=np.zeros((478, 3)),
        parsing=pr,
        photo_passes_gate=True,
        photo_issues=[],
    )


@pytest.fixture
def client():
    return TestClient(app)


def test_try_on_photo_endpoint(client):
    prepared = _prepared()
    fake_result = TryOnPhotoResult(
        original_b64="orig",
        generative=TryOnBranchResult(
            composite_b64="gen",
            categories={
                "makeup": TryOnCategoryMeta(renderer="generative", zones=["lips"]),
            },
        ),
        cv=None,
        active_mode="generative",
    )

    with (
        patch("app.services.try_on_service.prepare_face_bgr", return_value=prepared),
        patch("app.backends.try_on.engine.TryOnEngine.render_photo", return_value=fake_result),
    ):
        ok, buf = __import__("cv2").imencode(".jpg", prepared.image_bgr)
        assert ok
        meta = '{"season_twelve":"light_summer","categories":["makeup"],"generative":true}'
        res = client.post(
            "/try-on/photo",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
            data={"meta_json": meta},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["original_b64"] == "orig"
