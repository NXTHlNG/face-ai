from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from app.services.landmarks import LandmarkResult, dist_xy, landmark_xy
from app.services.mediapipe_vendor_contour import build_vendor_face_contour

_vendor_classifier_mod: Any = None

IDX_FOREHEAD_TOP_MP = 10
IDX_CHIN_MP = 152
IDX_LEFT_CHEEK_OUTER_MP = 234
IDX_RIGHT_CHEEK_OUTER_MP = 454
IDX_LEFT_JAW_MP = 176
IDX_RIGHT_JAW_MP = 400
IDX_LEFT_TEMPLE_MP = 103
IDX_RIGHT_TEMPLE_MP = 332


def _get_vendor_classifier_mod() -> Any:
    global _vendor_classifier_mod
    if _vendor_classifier_mod is None:
        root = Path(__file__).resolve().parents[2]
        path = root / "vendor" / "face_shape" / "face_shape_classifier.py"
        spec = importlib.util.spec_from_file_location("face_shape_classifier", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("face_shape_classifier: invalid import spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _vendor_classifier_mod = mod
    return _vendor_classifier_mod


def compute_geometry(lr: LandmarkResult) -> dict:
    lm = lr.landmarks_px
    if lm.shape[0] == 81:
        return _geometry_dlib81(lm)
    return _geometry_mediapipe(lr)


def _geometry_mediapipe(lr: LandmarkResult) -> dict:
    lm = lr.landmarks_px
    cheek_w = dist_xy(
        landmark_xy(lm, IDX_LEFT_CHEEK_OUTER_MP),
        landmark_xy(lm, IDX_RIGHT_CHEEK_OUTER_MP),
    )
    jaw_w = dist_xy(
        landmark_xy(lm, IDX_LEFT_JAW_MP),
        landmark_xy(lm, IDX_RIGHT_JAW_MP),
    )
    forehead_w = dist_xy(
        landmark_xy(lm, IDX_LEFT_TEMPLE_MP),
        landmark_xy(lm, IDX_RIGHT_TEMPLE_MP),
    )
    face_h = dist_xy(
        landmark_xy(lm, IDX_FOREHEAD_TOP_MP),
        landmark_xy(lm, IDX_CHIN_MP),
    )
    if cheek_w < 1e-6:
        cheek_w = 1e-6

    aspect_lm = face_h / cheek_w
    jaw_to_cheek_lm = jaw_w / cheek_w
    forehead_to_cheek_lm = forehead_w / cheek_w

    nose_bridge_y = float(lm[168, 1]) if lm.shape[0] > 168 else float(lm[:, 1].mean())
    split_top = abs(float(lm[IDX_FOREHEAD_TOP_MP, 1]) - nose_bridge_y) / max(face_h, 1e-6)
    split_bottom = abs(float(lm[IDX_CHIN_MP, 1]) - nose_bridge_y) / max(face_h, 1e-6)
    thirds_balance = 1.0 - min(1.0, abs(split_top - split_bottom))

    vendor_contour = build_vendor_face_contour(lr.image_rgb, lm)
    shape = "unknown"
    conf = 0.0
    cheek_norm = round(float(cheek_w / max(face_h, 1e-6)), 4)
    jaw_to_cheek = round(float(jaw_to_cheek_lm), 4)
    aspect = round(float(aspect_lm), 4)
    forehead_to_cheek = round(float(forehead_to_cheek_lm), 4)

    if vendor_contour is not None:
        try:
            mod = _get_vendor_classifier_mod()
            contour_f = vendor_contour.full_contour_px.astype(np.float32)
            m = mod.extract_metrics(contour_f, vendor_contour.forehead_top, vendor_contour.chin)
            label, pconf, diag = mod.classify_face_shape(m)
            shape = str(label)
            top_p = float(pconf.get(shape, 0.0))
            if bool(diag.get("low_confidence")):
                margin = float(diag.get("margin", 0.0))
                top_p = min(top_p, 0.38 + min(0.55, margin * 3.2))
            conf = round(max(0.0, min(1.0, top_p)), 4)
            fh = max(float(m["face_height"]), 1e-6)
            cheek_norm = round(float(m["w_cheek"]) / fh, 4)
            jaw_to_cheek = round(float(m["jaw_to_cheek"]), 4)
            aspect = round(float(m["h_to_w"]), 4)
            forehead_to_cheek = round(float(m["forehead_to_cheek"]), 4)
        except Exception:
            cheek_norm = round(float(cheek_w / max(face_h, 1e-6)), 4)
            jaw_to_cheek = round(float(jaw_to_cheek_lm), 4)
            aspect = round(float(aspect_lm), 4)
            forehead_to_cheek = round(float(forehead_to_cheek_lm), 4)
            shape, conf = _classify_face_shape(
                aspect_ratio=aspect_lm,
                jaw_to_cheek=jaw_to_cheek_lm,
                forehead_to_cheek=forehead_to_cheek_lm,
            )

    if shape == "unknown" and vendor_contour is None:
        shape, conf = _classify_face_shape(
            aspect_ratio=aspect_lm,
            jaw_to_cheek=jaw_to_cheek_lm,
            forehead_to_cheek=forehead_to_cheek_lm,
        )

    out: dict[str, Any] = {
        "cheekbone_width_norm": cheek_norm,
        "jaw_to_cheek_ratio": jaw_to_cheek,
        "face_aspect_ratio": aspect,
        "forehead_to_cheek_ratio": forehead_to_cheek,
        "vertical_thirds_balance": round(float(thirds_balance), 4),
        "face_shape": shape,
        "face_shape_confidence": conf,
        "geometry_landmark_set": "mediapipe_478",
        "vendor_face_contour_px": (
            vendor_contour.full_contour_px if vendor_contour is not None else None
        ),
    }
    return out


def _geometry_dlib81(lm: np.ndarray) -> dict:
    cheek_w = dist_xy(landmark_xy(lm, 2), landmark_xy(lm, 14))
    jaw_w = dist_xy(landmark_xy(lm, 3), landmark_xy(lm, 13))
    fh = lm[68:81]
    forehead_w = float(np.max(fh[:, 0]) - np.min(fh[:, 0]))
    top_y = float(np.min(fh[:, 1]))
    chin_y = float(lm[8, 1])
    face_h = abs(chin_y - top_y)
    if cheek_w < 1e-6:
        cheek_w = 1e-6

    aspect = face_h / cheek_w
    jaw_to_cheek = jaw_w / cheek_w
    forehead_to_cheek = forehead_w / cheek_w

    nose_y = float(lm[27, 1])
    split_top = abs(top_y - nose_y) / max(face_h, 1e-6)
    split_bottom = abs(nose_y - chin_y) / max(face_h, 1e-6)
    thirds_balance = 1.0 - min(1.0, abs(split_top - split_bottom))

    shape, conf = _classify_face_shape(
        aspect_ratio=aspect,
        jaw_to_cheek=jaw_to_cheek,
        forehead_to_cheek=forehead_to_cheek,
    )

    return {
        "cheekbone_width_norm": round(float(cheek_w / max(face_h, 1e-6)), 4),
        "jaw_to_cheek_ratio": round(float(jaw_to_cheek), 4),
        "face_aspect_ratio": round(float(aspect), 4),
        "forehead_to_cheek_ratio": round(float(forehead_to_cheek), 4),
        "vertical_thirds_balance": round(float(thirds_balance), 4),
        "face_shape": shape,
        "face_shape_confidence": round(float(conf), 4),
        "geometry_landmark_set": "dlib_iBUG_81",
    }


def _classify_face_shape(
    aspect_ratio: float,
    jaw_to_cheek: float,
    forehead_to_cheek: float,
) -> tuple[str, float]:
    scores: dict[str, float] = {}

    scores["oblong"] = max(0.0, min(1.0, (aspect_ratio - 1.35) / 0.35))
    scores["round"] = max(0.0, min(1.0, (1.22 - aspect_ratio) / 0.22)) * max(
        0.0,
        min(1.0, (jaw_to_cheek - 0.85) / 0.12),
    )
    scores["square"] = max(0.0, min(1.0, (jaw_to_cheek - 0.88) / 0.10)) * max(
        0.0,
        min(1.0, 1.5 - abs(aspect_ratio - 1.28) * 2),
    )
    scores["heart"] = max(0.0, min(1.0, (forehead_to_cheek - 1.02) / 0.08))
    scores["diamond"] = max(0.0, min(1.0, (1.05 - jaw_to_cheek) / 0.12)) * max(
        0.0,
        min(1.0, (forehead_to_cheek - 0.92) / 0.15),
    )
    oval_core = 1.0 - abs(aspect_ratio - 1.32) / 0.25
    scores["oval"] = max(0.0, min(1.0, oval_core)) * (
        1.0 - scores["oblong"] * 0.6 - scores["round"] * 0.5
    )

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    second = sorted(scores.values(), reverse=True)
    top = second[0]
    runner = second[1] if len(second) > 1 else 0.0
    confidence = max(0.0, min(1.0, (top - runner + 0.05) / 1.05))

    if top < 0.25:
        return "unknown", round(confidence * 0.5, 4)

    return best, round(confidence, 4)
