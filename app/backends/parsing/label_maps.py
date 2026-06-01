from __future__ import annotations

import numpy as np

from app.config import settings


def yakhyo_to_canonical(label_map: np.ndarray) -> dict[str, np.ndarray]:
    # yakhyo/face-parsing ONNX: 1=skin, 10=nose, 14=neck (CelebAMask-HQ layout).
    skin = np.isin(label_map, (1, 10, 14)).astype(np.uint8) * 255
    hair = (label_map == 17).astype(np.uint8) * 255
    brow = ((label_map == 2) | (label_map == 3)).astype(np.uint8) * 255
    glasses = (label_map == 6).astype(np.uint8) * 255
    lip = ((label_map == 12) | (label_map == 13)).astype(np.uint8) * 255
    eye = ((label_map == 4) | (label_map == 5)).astype(np.uint8) * 255
    return {
        "skin": skin,
        "hair": hair,
        "brow": brow,
        "glasses": glasses,
        "lip": lip,
        "eye": eye,
    }


def celebamask_to_canonical(label_map: np.ndarray) -> dict[str, np.ndarray]:
    skin = (label_map == 1).astype(np.uint8) * 255
    hair = (label_map == 13).astype(np.uint8) * 255
    brow = ((label_map == 6) | (label_map == 7)).astype(np.uint8) * 255
    glasses = (label_map == 3).astype(np.uint8) * 255
    lip = ((label_map == 11) | (label_map == 12)).astype(np.uint8) * 255
    eye = ((label_map == 4) | (label_map == 5)).astype(np.uint8) * 255
    return {
        "skin": skin,
        "hair": hair,
        "brow": brow,
        "glasses": glasses,
        "lip": lip,
        "eye": eye,
    }


# Official pyfacer FaRL label layouts (facer/face_parsing/farl.py).
FARL_OFFICIAL_LABEL_NAMES: dict[str, list[str]] = {
    "lapa/448": [
        "background",
        "face",
        "rb",
        "lb",
        "re",
        "le",
        "nose",
        "ulip",
        "imouth",
        "llip",
        "hair",
    ],
    "celebm/448": [
        "background",
        "neck",
        "face",
        "cloth",
        "rr",
        "lr",
        "rb",
        "lb",
        "re",
        "le",
        "nose",
        "imouth",
        "llip",
        "ulip",
        "hair",
        "eyeg",
        "hat",
        "earr",
        "neck_l",
    ],
}

# Map FaRL class names → canonical zones used by face-ai.
FARL_ZONE_SPECS: dict[str, dict[str, list[str]]] = {
    "lapa/448": {
        "skin": ["face", "nose"],
        "hair": ["hair"],
        "brow": ["rb", "lb"],
        "glasses": [],
        "lip": ["ulip", "imouth", "llip"],
        "eye": ["re", "le"],
    },
    "celebm/448": {
        "skin": ["face", "nose"],
        "hair": ["hair"],
        "brow": ["rb", "lb"],
        "glasses": ["eyeg"],
        "lip": ["imouth", "llip", "ulip"],
        "eye": ["re", "le"],
    },
}


def _farl_model_key(model_name: str) -> str:
    """``farl/celebm/448`` → ``celebm/448``."""
    parts = model_name.split("/", 1)
    return parts[1] if len(parts) == 2 else model_name


def farl_label_names_to_canonical(
    label_map: np.ndarray,
    label_names: list[str],
    *,
    zone_spec: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Build canonical zone masks from FaRL argmax labels + official class names."""
    name_to_idx = {name: idx for idx, name in enumerate(label_names)}

    def _mask_for(class_names: list[str]) -> np.ndarray:
        indices = [name_to_idx[name] for name in class_names if name in name_to_idx]
        if not indices:
            return np.zeros_like(label_map, dtype=np.uint8)
        return np.isin(label_map, indices).astype(np.uint8) * 255

    return {
        "skin": _mask_for(zone_spec["skin"]),
        "hair": _mask_for(zone_spec["hair"]),
        "brow": _mask_for(zone_spec["brow"]),
        "glasses": _mask_for(zone_spec["glasses"]),
        "lip": _mask_for(zone_spec["lip"]),
        "eye": _mask_for(zone_spec["eye"]),
    }


def farl_label_map_to_canonical(
    label_map: np.ndarray,
    model_name: str,
    label_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """FaRL argmax labels → canonical zones for ``farl/lapa/448`` or ``farl/celebm/448``."""
    model_key = _farl_model_key(model_name)
    zone_spec = FARL_ZONE_SPECS.get(model_key)
    if zone_spec is None:
        raise ValueError(f"Unsupported FaRL model: {model_name}")
    names = label_names or FARL_OFFICIAL_LABEL_NAMES[model_key]
    return farl_label_names_to_canonical(label_map, names, zone_spec=zone_spec)


def farl_lapa_label_map_to_canonical(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """Backward-compatible alias for ``farl/lapa/448``."""
    return farl_label_map_to_canonical(label_map, "farl/lapa/448")


def lapa_logits_to_canonical(
    logits: np.ndarray,
    *,
    skin_threshold: float = 0.5,
    model_name: str = "farl/lapa/448",
) -> dict[str, np.ndarray]:
    """FaRL logits → canonical zone masks via argmax."""
    del skin_threshold  # kept for config compat; zones use argmax, not prob threshold
    if logits.ndim == 3:
        label_map = np.argmax(logits, axis=0).astype(np.uint8)
    else:
        label_map = logits.astype(np.uint8)
    return farl_label_map_to_canonical(label_map, model_name)


# SegFace CelebAMask-HQ layout (network/models/segface_celeb.py, inference.py).
SEGFACE_CELEB_LABEL_NAMES: list[str] = [
    "background",
    "neck",
    "skin",
    "cloth",
    "l_ear",
    "r_ear",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "nose",
    "mouth",
    "l_lip",
    "u_lip",
    "hair",
    "eye_g",
    "hat",
    "ear_r",
    "neck_l",
]

# CelebAMask-HQ: class 2 is the face-skin PNG (*_skin.png); jaw/cheek sides are often class 1 (neck).
SEGFACE_CELEB_ZONE_SPEC: dict[str, list[str]] = {
    "skin": ["skin", "nose", "neck", "l_ear", "r_ear"],
    "hair": ["hair"],
    "brow": ["l_brow", "r_brow"],
    "glasses": ["eye_g"],
    "lip": ["mouth", "l_lip", "u_lip"],
    "eye": ["l_eye", "r_eye"],
}


def segface_to_canonical(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """CelebAMask-HQ 19-class layout used by SegFace (kartik-3004/segface)."""
    return farl_label_names_to_canonical(
        label_map,
        SEGFACE_CELEB_LABEL_NAMES,
        zone_spec=SEGFACE_CELEB_ZONE_SPEC,
    )


def canonical_to_parsing_result(
    zones: dict[str, np.ndarray],
    *,
    parsing_used: bool,
    label_map: np.ndarray | None,
    parsing_backend: str,
) -> "ParsingResult":
    from app.backends.parsing.types import ParsingResult

    skin = zones["skin"]
    hair = zones["hair"]
    brow = zones["brow"]
    union = np.clip(
        skin.astype(np.float32)
        + hair.astype(np.float32)
        + brow.astype(np.float32)
        + zones.get("eye", np.zeros_like(skin)).astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return ParsingResult(
        skin_mask=skin,
        hair_mask=hair,
        brow_mask=brow,
        eye_glass_mask=zones.get("glasses"),
        lip_mask=zones.get("lip"),
        eye_region_mask=zones.get("eye"),
        parsing_used=parsing_used,
        label_map=label_map,
        parsing_backend=parsing_backend,
        features_union_mask=union,
    )
