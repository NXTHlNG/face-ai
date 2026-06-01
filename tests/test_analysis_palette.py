import numpy as np

from app.backends.color.analysis_palette import (
    opencv_lab_to_rgb,
    rgb_to_hex,
    skin_rgb_hue_trim,
    swatch_from_rgb,
)


def test_rgb_to_hex():
    assert rgb_to_hex((220, 160, 140)) == "#DCA08C"


def test_opencv_lab_to_rgb_roundtrip():
    rgb = (180, 140, 120)
    lab = np.array([[list(rgb)]], dtype=np.uint8)
    import cv2

    lab_cv = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)[0, 0]
    out = opencv_lab_to_rgb((float(lab_cv[0]), float(lab_cv[1]), float(lab_cv[2])))
    assert all(abs(out[i] - rgb[i]) <= 2 for i in range(3))


def test_skin_rgb_hue_trim_matches_skin_tone():
    h, w = 80, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 120, 180)
    img[20:60, 20:60] = (200, 160, 120)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    rgb = skin_rgb_hue_trim(img, mask, min_pixels=8)
    assert rgb is not None
    assert rgb[0] > rgb[2]


def test_swatch_from_rgb_fields():
    sw = swatch_from_rgb("skin", (128, 64, 32))
    assert sw["hex"].startswith("#")
    assert len(sw["rgb"]) == 3
    assert len(sw["lab_opencv"]) == 3
    assert len(sw["lab_cielab"]) == 3


def test_season_reference_from_data():
    from app.backends.color.analysis_palette import _load_season_anchors

    anchors = _load_season_anchors()
    assert "spring" in anchors
    assert len(anchors["spring"]) == 3
