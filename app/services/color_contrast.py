from __future__ import annotations

import cv2
import numpy as np

from app.services.face_parsing import ParsingResult

_MP_LEFT_IRIS = (474, 475, 476, 477)
_MP_RIGHT_IRIS = (469, 470, 471, 472)
_MP_OUTER_LIP_RING = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
)


def _mean_lab(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    m = mask > 127
    if not np.any(m):
        return None
    roi = image_rgb[m].reshape(-1, 3)
    if roi.size == 0:
        return None
    bgr = cv2.cvtColor(roi.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return tuple(float(np.mean(bgr[:, i])) for i in range(3))


def _median_lab(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    m = mask > 127
    if not np.any(m):
        return None
    roi = image_rgb[m].reshape(-1, 3)
    if roi.size == 0:
        return None
    bgr = cv2.cvtColor(roi.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return tuple(float(np.median(bgr[:, i])) for i in range(3))


def _disk_mask(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    r: float,
) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = (xx - cx) ** 2 + (yy - cy) ** 2
    out = (d <= r * r).astype(np.uint8) * 255
    return out


def _iris_luminance_ring_mask(
    gray: np.ndarray,
    base_mask: np.ndarray,
    *,
    min_pixels: int = 16,
) -> np.ndarray | None:
    """Кольцо радужки: отрезаем зрачок (самые тёмные %) и склеру (самые светлые %)."""
    m = base_mask > 127
    if not np.any(m):
        return None
    vals = gray[m].astype(np.float32)
    if vals.size < 8:
        return None

    bands = (
        (20, 82),
        (14, 88),
        (10, 91),
        (8, 93),
    )
    g = gray.astype(np.float32)
    for lo_p, hi_p in bands:
        lo = float(np.percentile(vals, lo_p))
        hi = float(np.percentile(vals, hi_p))
        if hi <= lo + 2.0:
            continue
        sel = m & (g >= lo) & (g <= hi)
        if np.count_nonzero(sel) >= min_pixels:
            out = np.zeros(gray.shape[:2], dtype=np.uint8)
            out[sel] = 255
            return out
    return None


def _iris_saturation_trim(image_rgb: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Срезает почти нейтральные пиксели (зрачок/тени), не трогая при малой выборке."""
    if np.count_nonzero(ring > 127) < 28:
        return ring
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    rv = sat[ring > 127]
    if rv.size < 14:
        return ring
    lo_sat = float(np.percentile(rv, 10))
    lo_sat = min(lo_sat, 38.0)
    refined = (ring > 127) & (sat >= lo_sat)
    if np.count_nonzero(refined) < 12:
        return ring
    out = np.zeros_like(ring)
    out[refined] = 255
    return out


def _subtract_iris_pupil_blob(gray: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Убирает связный тёмный центр (зрачок), если он явно выделяется."""
    if np.count_nonzero(ring > 127) < 40:
        return ring
    m = ring > 127
    vals = gray[m].astype(np.float32)
    dark_thr = float(np.percentile(vals, 25))
    dark = m & (gray.astype(np.float32) <= dark_thr + 3.0)
    if np.count_nonzero(dark) < 8:
        return ring
    ring_area = float(np.count_nonzero(m))
    dark_u8 = dark.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
    if n <= 1:
        return ring
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return ring
    i = 1 + int(np.argmax(areas))
    blob_a = float(stats[i, cv2.CC_STAT_AREA])
    if blob_a < 6 or blob_a > max(28.0, ring_area * 0.42):
        return ring
    ys = stats[i, cv2.CC_STAT_TOP]
    xs = stats[i, cv2.CC_STAT_LEFT]
    ww = stats[i, cv2.CC_STAT_WIDTH]
    hh = stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = xs + ww / 2.0, ys + hh / 2.0
    r = float(np.sqrt(max(stats[i, cv2.CC_STAT_AREA], 1)) / np.pi) * 1.25
    r = min(r, min(gray.shape) * 0.08)
    h, w = gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    pupil_disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    kept = m & (~pupil_disk)
    if np.count_nonzero(kept) < 12:
        return ring
    out = np.zeros_like(ring)
    out[kept] = 255
    return out


def _median_lab_iris_ring(image_rgb: np.ndarray, base_mask: np.ndarray) -> tuple[float, float, float] | None:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ring = _iris_luminance_ring_mask(gray, base_mask)
    if ring is None:
        return None
    ring = _subtract_iris_pupil_blob(gray, ring)
    ring = _iris_saturation_trim(image_rgb, ring)
    return _median_lab(image_rgb, ring)


def _iris_lab_mediapipe(image_rgb: np.ndarray, lm: np.ndarray) -> tuple[float, float, float] | None:
    if lm.shape[0] < 478:
        return None
    h, w = image_rgb.shape[:2]
    labs: list[tuple[float, float, float]] = []
    for indices in (_MP_LEFT_IRIS, _MP_RIGHT_IRIS):
        pts = np.array([[lm[i, 0], lm[i, 1]] for i in indices], dtype=np.float32)
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        if not np.any(mask > 127):
            continue
        got = _median_lab_iris_ring(image_rgb, mask)
        if got:
            labs.append(got)
    if not labs:
        return None
    return tuple(float(np.mean([x[i] for x in labs])) for i in range(3))


def _iris_lab_ibug68(image_rgb: np.ndarray, lm: np.ndarray) -> tuple[float, float, float] | None:
    if lm.shape[0] < 68:
        return None
    h, w = image_rgb.shape[:2]
    labs: list[tuple[float, float, float]] = []
    for a, b in ((36, 39), (42, 45)):
        pta = lm[a, :2]
        ptb = lm[b, :2]
        ctr = (pta + ptb) / 2.0
        width = float(np.linalg.norm(pta - ptb))
        r = max(2.5, width * 0.17)
        disk = _disk_mask((h, w), float(ctr[0]), float(ctr[1]), r)
        if not np.any(disk > 127):
            continue
        got = _median_lab_iris_ring(image_rgb, disk)
        if got:
            labs.append(got)
    if not labs:
        return None
    return tuple(float(np.mean([x[i] for x in labs])) for i in range(3))


def _iris_lab_eye_region_mask(
    image_rgb: np.ndarray,
    eye_mask: np.ndarray,
    skin_mask: np.ndarray,
) -> tuple[float, float, float] | None:
    m = (eye_mask > 127) & (skin_mask <= 127)
    if not np.any(m):
        m = eye_mask > 127
    if not np.any(m):
        return None
    base = np.zeros(eye_mask.shape[:2], dtype=np.uint8)
    base[m] = 255
    return _median_lab_iris_ring(image_rgb, base)


def build_iris_ring_union_debug_mask(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray | None,
    pr: ParsingResult,
    glasses_ratio: float | None,
) -> np.ndarray | None:
    """Объединённая маска кольца радужки (для debug overlay), та же логика что и для LAB."""
    heavy = glasses_ratio is not None and glasses_ratio > 0.045
    lm = landmarks_px if landmarks_px is not None else np.empty((0, 3))
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    rings: list[np.ndarray] = []

    if not heavy and lm.shape[0] >= 478:
        for indices in (_MP_LEFT_IRIS, _MP_RIGHT_IRIS):
            pts = np.array([[lm[i, 0], lm[i, 1]] for i in indices], dtype=np.float32)
            hull = cv2.convexHull(pts)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
            ring = _iris_luminance_ring_mask(gray, mask)
            if ring is None:
                continue
            ring = _subtract_iris_pupil_blob(gray, ring)
            ring = _iris_saturation_trim(image_rgb, ring)
            rings.append(ring)

    if not rings and not heavy and lm.shape[0] >= 68:
        for a, b in ((36, 39), (42, 45)):
            pta = lm[a, :2]
            ptb = lm[b, :2]
            ctr = (pta + ptb) / 2.0
            width = float(np.linalg.norm(pta - ptb))
            r = max(2.5, width * 0.17)
            disk = _disk_mask((h, w), float(ctr[0]), float(ctr[1]), r)
            ring = _iris_luminance_ring_mask(gray, disk)
            if ring is None:
                continue
            ring = _subtract_iris_pupil_blob(gray, ring)
            ring = _iris_saturation_trim(image_rgb, ring)
            rings.append(ring)

    if not rings and pr.eye_region_mask is not None and np.any(pr.eye_region_mask > 127):
        m = (pr.eye_region_mask > 127) & (pr.skin_mask <= 127)
        if not np.any(m):
            m = pr.eye_region_mask > 127
        base = np.zeros((h, w), dtype=np.uint8)
        base[m] = 255
        ring = _iris_luminance_ring_mask(gray, base)
        if ring is not None:
            ring = _subtract_iris_pupil_blob(gray, ring)
            ring = _iris_saturation_trim(image_rgb, ring)
            rings.append(ring)

    if not rings:
        return None
    u = rings[0].astype(np.float32)
    for rk in rings[1:]:
        u = np.maximum(u, rk.astype(np.float32))
    return np.clip(u, 0, 255).astype(np.uint8)


def _lip_mask_from_landmarks(image_rgb: np.ndarray, lm: np.ndarray) -> np.ndarray | None:
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if lm.shape[0] >= 478:
        pts = []
        for i in _MP_OUTER_LIP_RING:
            if i < len(lm):
                pts.append([float(lm[i, 0]), float(lm[i, 1])])
        if len(pts) >= 3:
            arr = np.asarray(pts, dtype=np.float32)
            hull = cv2.convexHull(arr)
            poly = np.round(hull).astype(np.int32).reshape(-1, 2)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            cv2.fillConvexPoly(mask, poly, 255)
            return mask
    if lm.shape[0] >= 68:
        pts = [[float(lm[i, 0]), float(lm[i, 1])] for i in range(48, 60)]
        arr = np.asarray(pts, dtype=np.float32)
        hull = cv2.convexHull(arr)
        poly = np.round(hull).astype(np.int32).reshape(-1, 2)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        cv2.fillConvexPoly(mask, poly, 255)
        return mask
    return None


def _resolve_iris_lab(
    image_rgb: np.ndarray,
    pr: ParsingResult,
    landmarks_px: np.ndarray | None,
    glasses_ratio: float | None,
) -> tuple[float, float, float] | None:
    heavy_glasses = glasses_ratio is not None and glasses_ratio > 0.045
    lm = landmarks_px
    if lm is None:
        lm = np.empty((0, 3))
    if not heavy_glasses and lm.shape[0] >= 478:
        got = _iris_lab_mediapipe(image_rgb, lm)
        if got:
            return got
    if not heavy_glasses and lm.shape[0] >= 68:
        got = _iris_lab_ibug68(image_rgb, lm)
        if got:
            return got
    if pr.eye_region_mask is not None and np.any(pr.eye_region_mask > 127):
        got = _iris_lab_eye_region_mask(image_rgb, pr.eye_region_mask, pr.skin_mask)
        if got:
            return got
    return None


def _resolve_lip_lab(
    image_rgb: np.ndarray,
    pr: ParsingResult,
    landmarks_px: np.ndarray | None,
) -> tuple[float, float, float] | None:
    if pr.lip_mask is not None and np.any(pr.lip_mask > 127):
        got = _median_lab(image_rgb, pr.lip_mask)
        if got:
            return got
    if landmarks_px is not None:
        lm = _lip_mask_from_landmarks(image_rgb, landmarks_px)
        if lm is not None and np.any(lm > 127):
            got = _median_lab(image_rgb, lm)
            if got:
                return got
    return None


def _bucket_from_blend(peak: float, blend_support: float) -> str:
    effective = 0.52 * peak + 0.48 * blend_support
    if effective < 18.5:
        return "low"
    if effective < 34.0:
        return "medium"
    return "high"


def _value_contrast_index(
    d_hair: float | None,
    d_brow: float,
    d_iris: float | None,
    d_lip: float | None,
) -> float:
    caps = (48.0, 40.0, 36.0, 30.0)
    parts: list[float] = []
    weights: list[float] = []
    if d_hair is not None:
        parts.append(min(1.0, d_hair / caps[0]))
        weights.append(0.34)
    parts.append(min(1.0, d_brow / caps[1]))
    weights.append(0.28)
    if d_iris is not None:
        parts.append(min(1.0, d_iris / caps[2]))
        weights.append(0.28)
    if d_lip is not None:
        parts.append(min(1.0, d_lip / caps[3]))
        weights.append(0.10)
    sw = sum(weights)
    if sw <= 0:
        return 0.0
    wn = [w / sw for w in weights]
    composite = sum(p * w for p, w in zip(parts, wn))
    peak = max([x for x in (d_hair or 0.0, d_brow, d_iris or 0.0, d_lip or 0.0) if x >= 0])
    peak_n = min(1.0, peak / 46.0)
    score01 = 0.56 * peak_n + 0.44 * composite
    return round(float(score01 * 100.0), 2)


_EYE_LABEL_RU: dict[str, str] = {
    "blue": "голубые / синие",
    "gray": "серые",
    "green": "зелёные",
    "hazel": "ореховые",
    "amber": "янтарные",
    "light_brown": "светло-карие",
    "brown": "карие",
    "dark_brown": "тёмно-карие",
    "unknown": "не определено",
}


def _eye_color_hint(
    iris_lab: tuple[float, float, float] | None,
    glasses_pixel_ratio: float | None,
) -> tuple[str, str, float]:
    if iris_lab is None:
        return "unknown", _EYE_LABEL_RU["unknown"], 0.14

    gr = float(glasses_pixel_ratio or 0.0)
    if gr > 0.056:
        return "unknown", "не удалось оценить (очки)", max(0.15, 0.42 - gr * 4.0)

    L = float(iris_lab[0]) * 100.0 / 255.0
    aa = float(iris_lab[1]) - 128.0
    bb = float(iris_lab[2]) - 128.0
    chroma = float(np.hypot(aa, bb))

    def s_dark_brown() -> float:
        return max(0.0, (44.0 - L) / 44.0) * (1.1 + min(chroma, 28.0) / 55.0)

    def s_brown() -> float:
        mid = max(0.0, 1.0 - abs(L - 47.0) / 22.0)
        warm = max(0.0, (aa * 0.55 + bb * 0.65) / 22.0)
        return mid * (0.35 + warm)

    def s_light_brown() -> float:
        mid = max(0.0, 1.0 - abs(L - 54.0) / 18.0)
        warm = max(0.0, (aa + bb * 0.9) / 28.0)
        return mid * warm * 1.15

    def s_amber() -> float:
        return max(0.0, (bb - 2.0) / 18.0) * max(0.0, (L - 48.0) / 28.0) * max(0.0, (aa + 8.0) / 28.0)

    def s_hazel() -> float:
        spread = max(0.0, 1.0 - abs(L - 58.0) / 22.0)
        mix = max(0.0, aa / 22.0) * max(0.0, bb / 22.0)
        return spread * (0.25 + 0.45 * mix + 0.25 * min(chroma / 28.0, 1.2))

    def s_green() -> float:
        g_axis = max(0.0, (-aa - 1.5) / 16.0)
        return g_axis * max(0.0, (L - 42.0) / 35.0) * max(0.0, 1.0 - max(0.0, bb - 14.0) / 22.0)

    def s_blue() -> float:
        cool_b = max(0.0, (-bb - 3.0) / 18.0)
        mid_l = max(0.0, (L - 48.0) / 35.0)
        neutral_a = max(0.0, 1.0 - abs(aa) / 22.0)
        return cool_b * (0.45 + 0.55 * mid_l) * (0.65 + 0.35 * neutral_a)

    def s_gray() -> float:
        low_c = max(0.0, 1.0 - chroma / 18.0)
        light_v = max(0.0, (L - 54.0) / 28.0)
        return low_c * light_v * (1.1 + max(0.0, 12.0 - abs(bb)) / 40.0)

    scores = {
        "dark_brown": s_dark_brown(),
        "brown": s_brown(),
        "light_brown": s_light_brown(),
        "amber": s_amber(),
        "hazel": s_hazel(),
        "green": s_green(),
        "blue": s_blue(),
        "gray": s_gray(),
    }

    best = max(scores, key=scores.get)
    best_v = scores[best]
    second_v = sorted(scores.values(), reverse=True)[1]
    margin = best_v - second_v
    total = sum(scores.values()) + 1e-9

    if best_v < 0.08:
        return "unknown", _EYE_LABEL_RU["unknown"], 0.18

    conf = 0.34 + min(0.48, (best_v / total) * 2.8 + margin * 3.5)
    if chroma < 7.0 and L > 52:
        conf = min(0.82, conf + 0.06)
    if gr > 0.028:
        conf *= max(0.55, 1.0 - gr * 5.5)

    conf = float(min(0.86, max(0.22, conf)))
    return best, _EYE_LABEL_RU.get(best, best), conf


def compute_contrast_and_color(
    image_rgb: np.ndarray,
    pr: ParsingResult,
    landmarks_px: np.ndarray | None = None,
    glasses_pixel_ratio: float | None = None,
) -> tuple[dict, dict]:
    skin = _mean_lab(image_rgb, pr.skin_mask)
    brow = _mean_lab(image_rgb, pr.brow_mask)
    hair = _mean_lab(image_rgb, pr.hair_mask)

    skin_L = skin[0] if skin else 70.0
    brow_L = brow[0] if brow else skin_L
    hair_L = hair[0] if hair else None

    brow_delta = abs(brow_L - skin_L) if brow else 0.0
    hair_delta = abs(hair_L - skin_L) if hair_L is not None else None

    iris_lab = _resolve_iris_lab(image_rgb, pr, landmarks_px, glasses_pixel_ratio)
    iris_L = iris_lab[0] if iris_lab else None
    iris_delta = abs(iris_L - skin_L) if iris_L is not None else None
    if (
        glasses_pixel_ratio is not None
        and glasses_pixel_ratio > 0.035
        and iris_delta is not None
    ):
        iris_delta *= max(0.45, 1.0 - min(glasses_pixel_ratio * 8.0, 0.55))

    lip_lab = _resolve_lip_lab(image_rgb, pr, landmarks_px)
    lip_L = lip_lab[0] if lip_lab else None
    lip_delta = abs(lip_L - skin_L) if lip_L is not None else None

    d_list = [x for x in (hair_delta, brow_delta, iris_delta, lip_delta) if x is not None]
    blend_support = float(np.mean(d_list)) if d_list else brow_delta
    peak = max([x for x in (hair_delta or 0.0, brow_delta, iris_delta or 0.0, lip_delta or 0.0)])
    bucket = _bucket_from_blend(peak, blend_support)

    vci = _value_contrast_index(hair_delta, brow_delta, iris_delta, lip_delta)

    contrast = {
        "skin_L_mean": round(skin_L, 3),
        "brow_L_mean": round(brow_L, 3),
        "hair_L_mean": round(hair_L, 3) if hair_L is not None else None,
        "iris_L_mean": round(iris_L, 3) if iris_L is not None else None,
        "lip_L_mean": round(lip_L, 3) if lip_L is not None else None,
        "brow_skin_delta_L": round(float(brow_delta), 3),
        "hair_skin_delta_L": round(float(hair_delta), 3) if hair_delta is not None else None,
        "iris_skin_delta_L": round(float(iris_delta), 3) if iris_delta is not None else None,
        "lip_skin_delta_L": round(float(lip_delta), 3) if lip_delta is not None else None,
        "value_contrast_index": vci,
        "contrast_bucket": bucket,
        "parsing_used": pr.parsing_used,
    }

    skin_ab = (
        (float(skin[1]) - 128.0, float(skin[2]) - 128.0)
        if skin
        else (0.0, 0.0)
    )
    chroma = float(np.hypot(skin_ab[0], skin_ab[1]))

    hair_ab = None
    if hair:
        hair_ab = (float(hair[1]) - 128.0, float(hair[2]) - 128.0)

    iris_ab_color = None
    if iris_lab:
        iris_ab_color = (float(iris_lab[1]) - 128.0, float(iris_lab[2]) - 128.0)
    if iris_ab_color is None:
        approx = _iris_approx_ab(image_rgb, pr.skin_mask)
        if approx:
            iris_ab_color = approx

    undertone = _fused_undertone(skin_ab, hair_ab, iris_ab_color)

    hair_Lm = float(hair[0]) if hair else skin_L - 15.0
    depth = "medium"
    if skin_L > 71 and hair_Lm > 64:
        depth = "light"
    elif skin_L < 58 or hair_Lm < 48:
        depth = "deep"

    twelve, seasonal, s_conf = _infer_twelve_season(
        undertone,
        depth,
        chroma,
        bucket,
        vci,
        hair_ab,
        iris_ab_color,
        skin_L,
    )

    eye_cat, eye_label_ru, eye_conf = _eye_color_hint(iris_lab, glasses_pixel_ratio)

    color_features = {
        "skin_ab_mean": (round(skin_ab[0], 3), round(skin_ab[1], 3)),
        "skin_chroma_hint": round(chroma, 3),
        "hair_L_mean": round(hair_Lm, 3) if hair else None,
        "hair_ab_mean": (
            (round(hair_ab[0], 3), round(hair_ab[1], 3)) if hair_ab else None
        ),
        "iris_ab_mean": (
            (round(iris_ab_color[0], 3), round(iris_ab_color[1], 3))
            if iris_ab_color
            else None
        ),
        "eye_color_hint": eye_cat,
        "eye_color_label_ru": eye_label_ru,
        "eye_color_confidence": round(eye_conf, 4),
        "undertone_hint": undertone,
        "depth_hint": depth,
        "seasonal_twelve": twelve,
        "seasonal_twelve_confidence": s_conf,
        "seasonal_guess": seasonal,
        "seasonal_confidence": s_conf,
    }

    return contrast, color_features


def _fused_undertone(
    skin_ab: tuple[float, float],
    hair_ab: tuple[float, float] | None,
    iris_ab: tuple[float, float] | None,
) -> str:
    a_star, b_star = skin_ab
    warm_s = 0.0
    cool_s = 0.0
    if b_star > 4 and a_star > -2:
        warm_s += 2.2
    elif b_star < -4 or a_star < -6:
        cool_s += 2.2
    else:
        warm_s += 1.05
        cool_s += 1.05

    for ab, wt in ((hair_ab, 1.65), (iris_ab, 1.15)):
        if ab is None:
            continue
        _, hb = ab
        if hb > 3.2:
            warm_s += wt * min(max(hb, 0.0) / 13.5, 1.0)
        elif hb < -3.2:
            cool_s += wt * min(max(-hb, 0.0) / 13.5, 1.0)
        else:
            warm_s += wt * 0.38
            cool_s += wt * 0.38

    if warm_s > cool_s * 1.2:
        return "warm"
    if cool_s > warm_s * 1.2:
        return "cool"
    return "neutral"


def _clarity_axis(
    contrast_bucket: str,
    value_contrast_index: float,
    skin_chroma: float,
) -> tuple[float, float]:
    x = min(1.0, max(0.0, value_contrast_index / 74.0))
    if contrast_bucket == "high":
        x = min(1.0, x + 0.19)
    elif contrast_bucket == "low":
        x = max(0.0, x - 0.17)
    x = min(1.0, x + max(0.0, skin_chroma - 15.5) / 52.0)
    clear = min(1.0, max(0.0, x))
    soft = min(1.0, max(0.0, 1.0 - clear))
    if skin_chroma < 13.5 and contrast_bucket == "low":
        soft = min(1.0, soft + 0.14)
        clear = max(0.0, clear - 0.14)
    return clear, soft


def _warmth_axis_signed(
    undertone: str,
    hair_ab: tuple[float, float] | None,
    iris_ab: tuple[float, float] | None,
) -> float:
    if undertone == "warm":
        return 1.0
    if undertone == "cool":
        return -1.0
    acc = 0.0
    if hair_ab:
        acc += hair_ab[1]
    if iris_ab:
        acc += iris_ab[1]
    return float(np.clip(np.tanh(acc / 17.5), -1.0, 1.0))


def _depth_axis_signed(depth: str, skin_L: float) -> float:
    if depth == "light":
        return -1.0
    if depth == "deep":
        return 1.0
    t = (skin_L - 56.0) / (76.0 - 56.0)
    t = min(1.0, max(0.0, t))
    return float(1.0 - 2.0 * t)


TWELVE_TO_FOUR: dict[str, str] = {
    "light_spring": "spring",
    "true_spring": "spring",
    "bright_spring": "spring",
    "light_summer": "summer",
    "true_summer": "summer",
    "soft_summer": "summer",
    "soft_autumn": "autumn",
    "true_autumn": "autumn",
    "deep_autumn": "autumn",
    "deep_winter": "winter",
    "true_winter": "winter",
    "bright_winter": "winter",
}


def _gauss_1d(x: float, mu: float, sigma: float) -> float:
    s = max(sigma, 0.45)
    return float(np.exp(-0.5 * ((x - mu) / s) ** 2))


def _infer_twelve_season(
    undertone: str,
    depth: str,
    skin_chroma: float,
    contrast_bucket: str,
    value_contrast_index: float,
    hair_ab: tuple[float, float] | None,
    iris_ab: tuple[float, float] | None,
    skin_L: float,
) -> tuple[str, str, float]:
    clear, soft = _clarity_axis(contrast_bucket, value_contrast_index, skin_chroma)
    w = _warmth_axis_signed(undertone, hair_ab, iris_ab)
    d = _depth_axis_signed(depth, skin_L)

    wp = max(0.0, w)
    wn = max(0.0, -w)
    lp = max(0.0, -d)
    ln = max(0.0, d)

    low_c = contrast_bucket == "low"
    high_c = contrast_bucket == "high"
    vci_n = min(1.35, value_contrast_index / 76.0)
    chrom_n = min(1.0, skin_chroma / 36.0)

    gl = _gauss_1d(skin_L, 73.8, 5.2)
    gm = _gauss_1d(skin_L, 64.0, 7.8)
    gd = _gauss_1d(skin_L, 52.8, 6.6)

    win_mix = wn * (0.74 * ln + 0.26 * max(lp, 0.2))

    scores: dict[str, float] = {
        "light_spring": wp
        * lp
        * soft
        * (0.52 + 0.48 * gl)
        * (1.05 + 0.08 * (1.0 - chrom_n)),
        "bright_spring": wp * lp * clear * (0.42 + 0.58 * vci_n) * (1.08 if high_c else 1.0),
        "true_spring": wp
        * lp
        * (0.4 + 0.36 * clear + 0.38 * soft)
        * (0.85 + 0.35 * gm),
        "light_summer": wn
        * lp
        * soft
        * (0.5 + 0.5 * gl)
        * (1.06 - 0.12 * chrom_n),
        "soft_summer": wn
        * lp
        * (0.46 + 0.54 * soft)
        * (1.14 if low_c else 1.0)
        * (1.02 + 0.06 * (1.0 - vci_n)),
        "true_summer": wn
        * lp
        * (0.38 + 0.34 * clear + 0.44 * soft)
        * (0.88 + 0.32 * gm),
        "soft_autumn": wp * ln * (0.44 + 0.56 * soft) * (1.08 if low_c else 1.0),
        "true_autumn": wp
        * ln
        * (0.42 + 0.34 * clear + 0.38 * soft)
        * (0.88 + 0.28 * gm),
        "deep_autumn": wp * ln * (0.4 + 0.6 * gd) * (0.52 + 0.48 * ln) * (0.98 + 0.16 * soft),
        "bright_winter": win_mix * clear * (0.45 + 0.55 * vci_n)
        + wn * lp * clear * 0.26
        + (0.12 * wn * clear * high_c),
        "true_winter": wn
        * ln
        * (0.42 + 0.38 * clear + 0.34 * soft)
        * (0.86 + 0.3 * gm),
        "deep_winter": wn * ln * (0.42 + 0.58 * gd) * (0.48 + 0.52 * clear),
    }

    best_twelve = max(scores, key=scores.get)
    best_val = scores[best_twelve]
    total = sum(scores.values())
    ranked = sorted(scores.values(), reverse=True)
    second_val = ranked[1] if len(ranked) > 1 else 0.0
    margin = best_val - second_val
    rel = margin / max(total, 1e-9)

    conf = 0.34 + min(0.48, rel * 3.5 + margin * 3.0)
    if undertone == "neutral":
        conf *= 0.82
    if depth == "medium":
        conf *= 0.88
    if best_val / max(total, 1e-9) < 0.14:
        conf *= 0.68

    conf = float(min(0.88, max(0.22, conf)))
    if total < 1e-9 or best_val < 0.028:
        return "unknown", "unknown", max(0.17, conf * 0.48)

    four = TWELVE_TO_FOUR.get(best_twelve, "unknown")
    return best_twelve, four, conf


def _iris_approx_ab(image_rgb: np.ndarray, skin_mask: np.ndarray) -> tuple[float, float] | None:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    roi = gray[cy - h // 8 : cy + h // 16, cx - w // 6 : cx + w // 6]
    if roi.size == 0:
        return None
    _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = roi < (thr * 0.85)
    if not np.any(dark):
        return None
    patch = image_rgb[cy - h // 8 : cy + h // 16, cx - w // 6 : cx + w // 6]
    sel = patch[dark]
    if sel.size == 0:
        return None
    lab = cv2.cvtColor(sel.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return (float(np.mean(lab[:, 1]) - 128), float(np.mean(lab[:, 2]) - 128))


