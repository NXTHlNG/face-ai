import numpy as np

from app.backends.parsing.types import ParsingResult
from app.backends.try_on import mask_builder


def _parsing(h: int = 128, w: int = 128) -> ParsingResult:
    lip = np.zeros((h, w), dtype=np.uint8)
    lip[80:95, 50:78] = 255
    skin = np.zeros((h, w), dtype=np.uint8)
    skin[40:110, 30:98] = 255
    hair = np.zeros((h, w), dtype=np.uint8)
    hair[10:50, 30:98] = 255
    brow = np.zeros((h, w), dtype=np.uint8)
    brow[55:65, 40:88] = 255
    eye = np.zeros((h, w), dtype=np.uint8)
    eye[60:75, 38:90] = 255
    return ParsingResult(
        skin_mask=skin,
        hair_mask=hair,
        brow_mask=brow,
        eye_glass_mask=None,
        lip_mask=lip,
        eye_region_mask=eye,
        parsing_used=True,
        label_map=None,
        parsing_backend="test",
    )


def test_makeup_mask_nonzero_in_lip_region():
    pr = _parsing()
    mask, zones = mask_builder.build_makeup_mask(pr, (128, 128), None)
    assert "lips" in zones
    assert int(np.max(mask[82:93, 52:76])) > 0


def test_hairstyle_mask_matches_hair():
    pr = _parsing()
    mask = mask_builder.build_hairstyle_mask(pr, (128, 128))
    assert int(np.max(mask[15:45, 35:95])) > 0


def test_glasses_mask_with_landmarks():
    pr = _parsing()
    lm = np.zeros((478, 3), dtype=np.float64)
    lm[33, 0] = 50
    lm[33, 1] = 65
    lm[263, 0] = 78
    lm[263, 1] = 65
    lm[168, 0] = 64
    lm[168, 1] = 70
    mask = mask_builder.build_glasses_mask(pr, (128, 128), lm)
    assert int(np.max(mask)) > 0
