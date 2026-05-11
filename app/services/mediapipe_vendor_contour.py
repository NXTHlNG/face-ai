"""Контур лица и ключевые точки для vendor-классификатора формы (логика из vendor/face_shape/final_contour.py)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
LEFT_TEMPLE_HINTS = [54, 103, 67, 109]
RIGHT_TEMPLE_HINTS = [284, 332, 297, 338]

OLD_FOREHEAD_SEQ = [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251]
LOWER_FOREHEAD_SEQ = [71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301]


@dataclass
class VendorFaceContourResult:
    full_contour_px: np.ndarray
    forehead_top: np.ndarray
    chin: np.ndarray


def _lm_xy(lm: np.ndarray, idx: int) -> np.ndarray:
    return np.array(
        [float(lm[idx, 0]), float(lm[idx, 1])],
        dtype=np.float32,
    )


def _get_pts(lm: np.ndarray, ids: list[int]) -> list[np.ndarray]:
    return [_lm_xy(lm, i) for i in ids]


def _bezier_arc(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 30) -> list[np.ndarray]:
    pts = []
    for t in np.linspace(0, 1, n):
        pt = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
        pts.append(pt.astype(np.float32))
    return pts


def _refine_temple_on_boundary(
    gray: np.ndarray,
    center_pt: np.ndarray,
    temple_hint: np.ndarray,
    max_shift: float,
    samples: int = 45,
) -> np.ndarray:
    c = np.array(center_pt, dtype=np.float32)
    hint = np.array(temple_hint, dtype=np.float32)
    vec = hint - c
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return hint
    direction = vec / dist
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    best_pt = hint.copy()
    best_score = -1.0
    for t in np.linspace(0.72, 1.18, samples):
        p = c + direction * dist * t
        x = int(np.clip(round(p[0]), 0, gray.shape[1] - 1))
        y = int(np.clip(round(p[1]), 0, gray.shape[0] - 1))
        g = float(grad[y, x])
        distance_penalty = abs(t - 1.0) * 15.0
        score = g - distance_penalty
        if score > best_score:
            best_score = score
            best_pt = np.array([x, y], dtype=np.float32)
    shift = best_pt - hint
    shift_len = float(np.linalg.norm(shift))
    if shift_len > max_shift:
        best_pt = hint + shift * (max_shift / (shift_len + 1e-6))
    return best_pt


def _estimate_forehead(
    image_rgb: np.ndarray,
    face_oval: list[np.ndarray],
    left_brow: list[np.ndarray],
    right_brow: list[np.ndarray],
    lm: np.ndarray,
) -> dict[str, np.ndarray]:
    chin = max(face_oval, key=lambda p: float(p[1]))
    left_temple_hint = min(_get_pts(lm, LEFT_TEMPLE_HINTS), key=lambda p: float(p[0]))
    right_temple_hint = max(_get_pts(lm, RIGHT_TEMPLE_HINTS), key=lambda p: float(p[0]))

    brow_center = np.mean(np.array(left_brow + right_brow), axis=0)
    temple_mid = (left_temple_hint + right_temple_hint) / 2.0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    temple_width_hint = float(np.linalg.norm(right_temple_hint - left_temple_hint))
    max_shift = max(6.0, temple_width_hint * 0.09)

    left_temple = _refine_temple_on_boundary(
        gray=gray,
        center_pt=temple_mid,
        temple_hint=left_temple_hint,
        max_shift=max_shift,
    )
    right_temple = _refine_temple_on_boundary(
        gray=gray,
        center_pt=temple_mid,
        temple_hint=right_temple_hint,
        max_shift=max_shift,
    )

    face_height = float(chin[1] - temple_mid[1])
    forehead_raise = face_height * 0.32

    forehead_top = np.array(
        [temple_mid[0], max(0.0, float(temple_mid[1] - forehead_raise))],
        dtype=np.float32,
    )

    left_forehead = left_temple.astype(np.float32)
    right_forehead = right_temple.astype(np.float32)
    _bezier_arc(left_forehead, forehead_top, right_forehead, n=25)

    return {
        "left_temple": left_forehead,
        "right_temple": right_forehead,
        "forehead_top": forehead_top,
        "chin": np.array(chin, dtype=np.float32),
    }


def _build_face_oval_with_upper_row(
    face_oval_pts: list[np.ndarray],
    lm: np.ndarray,
    forehead_top: np.ndarray,
) -> np.ndarray:
    pts = np.array(face_oval_pts, dtype=np.float32)

    if len(OLD_FOREHEAD_SEQ) != len(LOWER_FOREHEAD_SEQ):
        raise ValueError("OLD_FOREHEAD_SEQ и LOWER_FOREHEAD_SEQ должны быть одинаковой длины")

    old_pts = []
    low_pts = []
    for old_idx, low_idx in zip(OLD_FOREHEAD_SEQ, LOWER_FOREHEAD_SEQ):
        old_pts.append(_lm_xy(lm, old_idx))
        low_pts.append(_lm_xy(lm, low_idx))
    old_pts = np.array(old_pts, dtype=np.float32)
    low_pts = np.array(low_pts, dtype=np.float32)

    up1_pts = 2 * old_pts - low_pts

    try:
        idx10_in_old_seq = OLD_FOREHEAD_SEQ.index(10)
    except ValueError:
        idx10_in_old_seq = None

    if idx10_in_old_seq is not None:
        p10_up1 = up1_pts[idx10_in_old_seq]
        dy = p10_up1[1] - forehead_top[1]
        up2_pts = up1_pts.copy()
        up2_pts[:, 1] = np.clip(up2_pts[:, 1] - dy, 0, np.max(pts[:, 1]))
    else:
        up2_pts = up1_pts

    squeeze_k = 0.05
    min_x = float(np.min(up2_pts[:, 0]))
    max_x = float(np.max(up2_pts[:, 0]))
    center_x = (min_x + max_x) / 2.0
    width = max_x - min_x
    shift0 = width * squeeze_k
    shift1 = shift0
    shift2 = shift0 * 0.5
    n = len(up2_pts)
    left_idx0, right_idx0 = 0, n - 1
    left_idx1, right_idx1 = 1, n - 2
    left_idx2, right_idx2 = 2, n - 3

    up2_pts[left_idx0, 0] = min(center_x, up2_pts[left_idx0, 0] + shift0)
    up2_pts[right_idx0, 0] = max(center_x, up2_pts[right_idx0, 0] - shift0)
    up2_pts[left_idx1, 0] = min(center_x, up2_pts[left_idx1, 0] + shift1)
    up2_pts[right_idx1, 0] = max(center_x, up2_pts[right_idx1, 0] - shift1)
    up2_pts[left_idx2, 0] = min(center_x, up2_pts[left_idx2, 0] + shift2)
    up2_pts[right_idx2, 0] = max(center_x, up2_pts[right_idx2, 0] - shift2)

    result_pts: list[list[float]] = []
    old_set = set(OLD_FOREHEAD_SEQ)
    inserted = False

    for idx in FACE_OVAL:
        if idx in old_set:
            if not inserted:
                for p in up2_pts:
                    result_pts.append(p.tolist())
                inserted = True
            continue
        result_pts.append(pts[FACE_OVAL.index(idx)].tolist())

    return np.array(result_pts, dtype=np.float32)


def _build_ordered_face_contour(face_oval_pts: np.ndarray) -> np.ndarray:
    pts = np.array(face_oval_pts, dtype=np.float32)
    center_x = float(np.mean(pts[:, 0]))
    chin_idx = int(np.argmax(pts[:, 1]))

    left_side = pts[pts[:, 0] <= center_x]
    right_side = pts[pts[:, 0] > center_x]

    left_side = sorted(left_side.tolist(), key=lambda p: p[1])
    right_side = sorted(right_side.tolist(), key=lambda p: p[1])

    chin = pts[chin_idx].tolist()

    if not left_side or left_side[-1] != chin:
        left_side.append(chin)
    if not right_side or right_side[-1] != chin:
        right_side.append(chin)

    contour: list[list[float]] = []
    contour.extend(left_side)
    contour.extend(right_side[::-1][1:])

    return np.array(contour, dtype=np.int32)


def build_vendor_face_contour(image_rgb: np.ndarray, lm: np.ndarray) -> VendorFaceContourResult | None:
    if lm.shape[0] < max(FACE_OVAL) + 1:
        return None

    face_oval = _get_pts(lm, FACE_OVAL)
    left_brow = _get_pts(lm, LEFT_BROW)
    right_brow = _get_pts(lm, RIGHT_BROW)

    forehead = _estimate_forehead(image_rgb, face_oval, left_brow, right_brow, lm)

    face_oval_modified = _build_face_oval_with_upper_row(
        face_oval_pts=face_oval,
        lm=lm,
        forehead_top=forehead["forehead_top"],
    )

    full_contour = _build_ordered_face_contour(face_oval_modified)

    return VendorFaceContourResult(
        full_contour_px=full_contour,
        forehead_top=forehead["forehead_top"],
        chin=forehead["chin"],
    )
