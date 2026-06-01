import numpy as np

from app.backends.color.aua_heuristics import (
    aggregate_chroma,
    chroma_from_ab,
    lip_lab_brightness_clusters,
    skin_lab_hue_trim,
)


def test_skin_hue_trim_prefers_yellow_range():
    h, w = 80, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 120, 180)  # bluish (low hue)
    img[20:60, 20:60] = (200, 160, 120)  # skin-like (hue ~13–24 in HSV)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    lab = skin_lab_hue_trim(img, mask, min_pixels=8)
    assert lab is not None


def test_lip_brightness_clusters_middle_tier():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    for i, lum in enumerate((30, 80, 200)):
        img[12 + i * 5, 15, :] = (int(lum * 0.9), int(lum * 0.4), int(lum * 0.5))
    lab = lip_lab_brightness_clusters(img, mask, min_pixels=3)
    assert lab is not None


def test_aggregate_chroma_includes_zones():
    skin = (10.0, 10.0)
    eyes = (20.0, 0.0)
    lips = (0.0, 20.0)
    agg = aggregate_chroma(skin, eyes, lips)
    solo = chroma_from_ab(skin)
    assert agg > solo
