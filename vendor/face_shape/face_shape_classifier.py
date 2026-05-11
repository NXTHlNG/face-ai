import argparse
import json
from typing import Dict, List, Tuple

import numpy as np


def load_geometry(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contour = np.array(data["full_face_contour"], dtype=np.float32)
    forehead_top = np.array(data["forehead_top"], dtype=np.float32)
    chin = np.array(data["chin"], dtype=np.float32)
    return contour, forehead_top, chin


def _width_at_relative_height(
    xs: np.ndarray, ys: np.ndarray, top_y: float, height: float, rel: float, band_rel: float = 0.04
) -> float:
    target_y = top_y + rel * height
    for mult in (1.0, 2.0, 4.0):
        band = max(2.0, height * band_rel * mult)
        mask = (ys >= target_y - band) & (ys <= target_y + band)
        if np.any(mask):
            row = xs[mask]
            return float(np.max(row) - np.min(row))
    dy = np.abs(ys - target_y)
    k = min(len(xs), max(8, len(xs) // 5))
    idx = np.argpartition(dy, k - 1)[:k]
    row = xs[idx]
    return float(np.max(row) - np.min(row))


def _jaw_angle_at_level(pts: np.ndarray, chin: np.ndarray, face_height: float, y_frac_from_chin: float) -> float:
    """Угол при подбородке между лучами chin→левая и chin→правая точки контура на уровне y."""
    target_y = float(chin[1] - y_frac_from_chin * face_height)
    y_band = max(4.0, 0.035 * face_height)
    y_mask = (pts[:, 1] >= target_y - y_band) & (pts[:, 1] <= target_y + y_band)
    pool = pts[y_mask]
    if len(pool) < 4:
        pool = pts

    cx_margin = max(3.0, 0.008 * face_height)
    left_pool = pool[pool[:, 0] < chin[0] - cx_margin]
    right_pool = pool[pool[:, 0] > chin[0] + cx_margin]
    if len(left_pool) == 0 or len(right_pool) == 0:
        return 0.0

    left_jaw = left_pool[np.argmin(np.abs(left_pool[:, 1] - target_y))]
    right_jaw = right_pool[np.argmin(np.abs(right_pool[:, 1] - target_y))]

    vl = left_jaw - chin
    vr = right_jaw - chin
    nl = np.linalg.norm(vl)
    nr = np.linalg.norm(vr)
    if nl < 1e-6 or nr < 1e-6:
        return 0.0

    cos_theta = float(np.dot(vl, vr) / (nl * nr))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _jaw_angle_proxy(pts: np.ndarray, chin: np.ndarray, face_height: float) -> float:
    fracs = [0.10, 0.14, 0.18]
    angles = [_jaw_angle_at_level(pts, chin, face_height, f) for f in fracs]
    angles = [a for a in angles if a > 1e-6]
    if not angles:
        return 0.0
    return float(np.mean(angles))


def _nearest_point(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return target.copy()
    idx = int(np.argmin(np.linalg.norm(points - target, axis=1)))
    return points[idx]


def _chin_roundness(pts: np.ndarray, chin: np.ndarray, face_height: float) -> float:
    sample_y = float(chin[1] - 0.10 * face_height)
    y_band = max(4.0, 0.035 * face_height)
    band = pts[(pts[:, 1] >= sample_y - y_band) & (pts[:, 1] <= sample_y + y_band)]
    if len(band) < 4:
        return 0.5

    left = band[band[:, 0] < chin[0] - 2.0]
    right = band[band[:, 0] > chin[0] + 2.0]
    if len(left) == 0 or len(right) == 0:
        return 0.5

    left_pt = _nearest_point(left, np.array([chin[0] - 0.08 * face_height, sample_y], dtype=np.float32))
    right_pt = _nearest_point(right, np.array([chin[0] + 0.08 * face_height, sample_y], dtype=np.float32))

    vl = left_pt - chin
    vr = right_pt - chin
    nl = np.linalg.norm(vl)
    nr = np.linalg.norm(vr)
    if nl < 1e-6 or nr < 1e-6:
        return 0.5

    cos_theta = float(np.dot(vl, vr) / (nl * nr))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    # Нормируем: острый подбородок -> ближе к 0, мягкий/круглый -> ближе к 1.
    return float(np.clip((angle_deg - 72.0) / 70.0, 0.0, 1.0))


def _jaw_slope_balance(pts: np.ndarray, chin: np.ndarray, face_height: float) -> float:
    y1 = float(chin[1] - 0.12 * face_height)
    y2 = float(chin[1] - 0.24 * face_height)
    b1 = max(3.0, 0.03 * face_height)
    b2 = max(4.0, 0.04 * face_height)

    band1 = pts[(pts[:, 1] >= y1 - b1) & (pts[:, 1] <= y1 + b1)]
    band2 = pts[(pts[:, 1] >= y2 - b2) & (pts[:, 1] <= y2 + b2)]
    if len(band1) < 4 or len(band2) < 4:
        return 0.0

    l1 = _nearest_point(band1[band1[:, 0] < chin[0]], np.array([chin[0] - 0.1 * face_height, y1], dtype=np.float32))
    l2 = _nearest_point(band2[band2[:, 0] < chin[0]], np.array([chin[0] - 0.12 * face_height, y2], dtype=np.float32))
    r1 = _nearest_point(band1[band1[:, 0] > chin[0]], np.array([chin[0] + 0.1 * face_height, y1], dtype=np.float32))
    r2 = _nearest_point(band2[band2[:, 0] > chin[0]], np.array([chin[0] + 0.12 * face_height, y2], dtype=np.float32))

    left_dx = float(l2[0] - l1[0])
    left_dy = float(l2[1] - l1[1])
    right_dx = float(r2[0] - r1[0])
    right_dy = float(r2[1] - r1[1])
    if abs(left_dy) < 1e-6 or abs(right_dy) < 1e-6:
        return 0.0

    left_slope = abs(left_dx / left_dy)
    right_slope = abs(right_dx / right_dy)
    return float(abs(left_slope - right_slope))


def _lower_face_width_curve(
    xs: np.ndarray, ys: np.ndarray, top_y: float, face_height: float
) -> Tuple[float, List[float]]:
    rels = [0.62, 0.70, 0.78, 0.86]
    widths = [
        _width_at_relative_height(xs, ys, top_y, face_height, rel=rel, band_rel=0.03)
        for rel in rels
    ]
    if widths[0] <= 1e-6:
        return 0.0, widths
    # Рост ширины вниз по лицу (+) / сужение к подбородку (-).
    curve = (widths[-1] - widths[0]) / (widths[0] + 1e-6)
    return float(curve), [float(w) for w in widths]


def extract_metrics(contour: np.ndarray, forehead_top: np.ndarray, chin: np.ndarray) -> Dict[str, float]:
    pts = np.array(contour, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]

    top_y = float(min(forehead_top[1], np.min(ys)))
    bottom_y = float(max(chin[1], np.max(ys)))
    face_height = max(1.0, bottom_y - top_y)

    w_top = _width_at_relative_height(xs, ys, top_y, face_height, rel=0.10)
    w_forehead = _width_at_relative_height(xs, ys, top_y, face_height, rel=0.22)
    if w_forehead < 1e-6:
        w_forehead = max(
            w_top,
            _width_at_relative_height(xs, ys, top_y, face_height, rel=0.15),
            _width_at_relative_height(xs, ys, top_y, face_height, rel=0.18),
        )
    w_cheek = _width_at_relative_height(xs, ys, top_y, face_height, rel=0.45)
    w_upper_jaw = _width_at_relative_height(xs, ys, top_y, face_height, rel=0.68)
    w_jaw = _width_at_relative_height(xs, ys, top_y, face_height, rel=0.80)

    max_w = max(w_top, w_forehead, w_cheek, w_upper_jaw, w_jaw, 1e-6)
    h_to_w = face_height / max_w
    height_width_delta = abs(face_height - max_w) / max(face_height, max_w, 1e-6)
    forehead_to_cheek = w_forehead / (w_cheek + 1e-6)
    jaw_to_cheek = w_jaw / (w_cheek + 1e-6)
    taper = (w_forehead - w_jaw) / (w_cheek + 1e-6)
    jaw_angle = _jaw_angle_proxy(pts, chin, face_height)
    chin_roundness = _chin_roundness(pts, chin, face_height)
    jaw_slope_balance = _jaw_slope_balance(pts, chin, face_height)
    lower_face_width_curve, lower_widths = _lower_face_width_curve(xs, ys, top_y, face_height)

    return {
        "face_height": float(face_height),
        "w_top": float(w_top),
        "w_forehead": float(w_forehead),
        "w_cheek": float(w_cheek),
        "w_upper_jaw": float(w_upper_jaw),
        "w_jaw": float(w_jaw),
        "h_to_w": float(h_to_w),
        "height_width_delta": float(height_width_delta),
        "forehead_to_cheek": float(forehead_to_cheek),
        "jaw_to_cheek": float(jaw_to_cheek),
        "taper": float(taper),
        "jaw_angle": float(jaw_angle),
        "chin_roundness": float(chin_roundness),
        "jaw_slope_balance": float(jaw_slope_balance),
        "lower_face_width_curve": float(lower_face_width_curve),
        "lower_width_62": lower_widths[0],
        "lower_width_70": lower_widths[1],
        "lower_width_78": lower_widths[2],
        "lower_width_86": lower_widths[3],
    }


def _peak_score(value: float, center: float, half_span: float) -> float:
    if half_span <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value - center) / half_span)


def classify_face_shape(metrics: Dict[str, float]) -> Tuple[str, Dict[str, float], Dict[str, object]]:
    h_to_w = metrics["h_to_w"]
    hw_delta = metrics["height_width_delta"]
    f_to_c = metrics["forehead_to_cheek"]
    j_to_c = metrics["jaw_to_cheek"]
    taper = metrics["taper"]
    jaw_angle = metrics["jaw_angle"]
    chin_roundness = metrics["chin_roundness"]
    jaw_slope_balance = metrics["jaw_slope_balance"]
    lower_curve = metrics["lower_face_width_curve"]

    scores = {}

    scores["oval"] = (
        0.33 * _peak_score(h_to_w, 1.42, 0.28)
        + 0.24 * _peak_score(f_to_c, 0.98, 0.16)
        + 0.20 * _peak_score(j_to_c, 0.90, 0.16)
        + 0.12 * _peak_score(jaw_angle, 100.0, 16.0)
        + 0.11 * _peak_score(chin_roundness, 0.55, 0.22)
    )

    scores["round"] = (
        0.28 * _peak_score(h_to_w, 1.10, 0.20)
        + 0.20 * _peak_score(f_to_c, 1.00, 0.14)
        + 0.16 * _peak_score(j_to_c, 0.95, 0.16)
        + 0.18 * _peak_score(jaw_angle, 118.0, 18.0)
        + 0.18 * _peak_score(chin_roundness, 0.62, 0.38)
        + 0.10 * _peak_score(lower_curve, -0.12, 0.20)
    )

    # Квадрат/прямоугольник: широкий допуск по углу и по chin_roundness (узкий пик 0.22 давал почти 0 при типичных ~0.45–0.55).
    scores["square"] = (
        0.22 * _peak_score(h_to_w, 1.22, 0.22)
        + 0.20 * _peak_score(f_to_c, 1.00, 0.16)
        + 0.20 * _peak_score(j_to_c, 1.02, 0.16)
        + 0.22 * _peak_score(jaw_angle, 108.0, 28.0)
        + 0.16 * _peak_score(chin_roundness, 0.38, 0.42)
    )

    scores["heart"] = (
        0.32 * _peak_score(f_to_c, 1.12, 0.16)
        + 0.26 * _peak_score(j_to_c, 0.84, 0.14)
        + 0.18 * _peak_score(taper, 0.24, 0.18)
        + 0.14 * _peak_score(chin_roundness, 0.32, 0.22)
        + 0.10 * _peak_score(h_to_w, 1.32, 0.26)
    )

    scores["triangle"] = (
        0.32 * _peak_score(j_to_c, 1.12, 0.16)
        + 0.26 * _peak_score(f_to_c, 0.88, 0.14)
        + 0.20 * _peak_score(taper, -0.20, 0.18)
        + 0.12 * _peak_score(lower_curve, 0.14, 0.18)
        + 0.10 * _peak_score(h_to_w, 1.22, 0.24)
    )

    # При hw_delta <= 0.24 — только round vs square (овалы с похожими H/W попадают сюда).
    # При hw_delta > 0.24 — круг обнуляем, квадрат не обнуляем (конкурирует с oval/heart/triangle).
    is_round_square_gate = hw_delta <= 0.24
    shape_gate = "round_square" if is_round_square_gate else "other_shapes"
    if is_round_square_gate:
        scores["oval"] = 0.0
        scores["heart"] = 0.0
        scores["triangle"] = 0.0
        # Разделение только внутри пары round/square.
        if chin_roundness >= 0.62 or jaw_angle >= 118.0:
            scores["round"] *= 1.18
            scores["square"] *= 0.92
        elif chin_roundness <= 0.48 or jaw_angle <= 112.0:
            scores["square"] *= 1.22
            scores["round"] *= 0.88
    else:
        scores["round"] = 0.0
        # Узкий буст square: не для плавной челюсти (>=115°) и не при мягком подбородке.
        if (
            h_to_w >= 1.18
            and h_to_w <= 1.52
            and 95.0 <= jaw_angle < 115.0
            and chin_roundness < 0.45
        ):
            scores["square"] *= 1.12

    oval_profile_ok = (
        1.28 <= h_to_w <= 1.58
        and 0.88 <= f_to_c <= 1.05
        and 0.80 <= j_to_c <= 0.98
        and 88.0 <= jaw_angle <= 114.0
    )
    if not oval_profile_ok:
        scores["oval"] *= 0.72

    if f_to_c > 1.08 and j_to_c < 0.92 and taper > 0.10:
        scores["heart"] *= 1.12
        scores["triangle"] *= 0.88
        scores["oval"] *= 0.72
    if j_to_c > 1.08 and f_to_c < 0.94 and lower_curve > 0.02:
        scores["triangle"] *= 1.12
        scores["heart"] *= 0.88
        scores["oval"] *= 0.70
    if j_to_c > 0.97 and chin_roundness < 0.45 and jaw_angle < 108.0 and is_round_square_gate:
        scores["square"] *= 1.18
        scores["oval"] *= 0.68
    if chin_roundness > 0.78 and jaw_angle > 118.0 and h_to_w < 1.25 and is_round_square_gate:
        scores["round"] *= 1.12
        scores["oval"] *= 0.72
    if j_to_c > 0.98 and jaw_angle < 108.0 and is_round_square_gate:
        scores["square"] *= 1.20
        scores["oval"] *= 0.66
    if taper > 0.18 and f_to_c > 1.03 and j_to_c < 0.90:
        scores["heart"] *= 1.15
        scores["oval"] *= 0.66
    if taper < -0.10 and j_to_c > 0.95 and f_to_c < 0.95:
        scores["triangle"] *= 1.15
        scores["oval"] *= 0.66
    if jaw_slope_balance > 0.28:
        scores["square"] *= 0.96
        scores["triangle"] *= 0.92
    if h_to_w > 1.55 and is_round_square_gate:
        scores["round"] *= 0.80
        scores["square"] *= 0.90

    total = sum(scores.values()) + 1e-9
    confidence = {k: float(v / total) for k, v in scores.items()}
    ranked = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
    best = ranked[0][0]
    second = ranked[1][0]
    margin = float(ranked[0][1] - ranked[1][1])

    low_confidence = margin < 0.08

    evidence = []
    if is_round_square_gate:
        evidence.append("shape_gate=round_square (hw_delta<=0.24): только round vs square")
    else:
        evidence.append("shape_gate=other_shapes (hw_delta>0.24): round=0, square конкурирует с oval/heart/triangle")
    if j_to_c > 1.05 and f_to_c < 0.95:
        evidence.append("челюсть шире лба")
    if f_to_c > 1.05 and j_to_c < 0.92:
        evidence.append("лоб шире челюсти")
    if chin_roundness > 0.75:
        evidence.append("мягкий подбородок")
    elif chin_roundness < 0.35:
        evidence.append("угловатый подбородок")
    if jaw_angle < 98:
        evidence.append("выраженный угол челюсти")
    if jaw_angle > 118:
        evidence.append("плавная линия челюсти")
    if not oval_profile_ok:
        evidence.append("профиль овала не подтвержден")
    if not evidence:
        evidence.append("признаки смешанного типа")

    diagnostics = {
        "shape_gate": shape_gate,
        "secondary_candidate": second,
        "margin": margin,
        "low_confidence": low_confidence,
        "evidence": evidence,
    }
    return best, confidence, diagnostics


def run(json_path: str) -> Dict[str, object]:
    contour, forehead_top, chin = load_geometry(json_path)
    metrics = extract_metrics(contour, forehead_top, chin)
    label, confidence, diagnostics = classify_face_shape(metrics)
    return {
        "label": label,
        "confidence": confidence,
        "metrics": metrics,
        "shape_gate": diagnostics["shape_gate"],
        "secondary_candidate": diagnostics["secondary_candidate"],
        "margin": diagnostics["margin"],
        "low_confidence": diagnostics["low_confidence"],
        "evidence": diagnostics["evidence"],
    }


def main():
    parser = argparse.ArgumentParser(description="Face shape classifier from contour JSON.")
    parser.add_argument(
        "--json",
        default="D:/face_points_fixed_combined.json",
        help="Path to JSON file with full_face_contour, forehead_top and chin.",
    )
    args = parser.parse_args()

    result = run(args.json)
    metrics = result["metrics"]
    confidence = result["confidence"]

    print(f"Файл: {args.json}")
    print("Ключевые метрики:")
    print(f"  height: {metrics['face_height']:.2f}")
    print(f"  h_to_w: {metrics['h_to_w']:.3f}")
    print(f"  height_width_delta: {metrics['height_width_delta']:.3f}")
    print(f"  forehead_to_cheek: {metrics['forehead_to_cheek']:.3f}")
    print(f"  jaw_to_cheek: {metrics['jaw_to_cheek']:.3f}")
    print(f"  taper: {metrics['taper']:.3f}")
    print(f"  jaw_angle: {metrics['jaw_angle']:.2f}")
    print(f"  chin_roundness: {metrics['chin_roundness']:.3f}")
    print(f"  jaw_slope_balance: {metrics['jaw_slope_balance']:.3f}")
    print(f"  lower_face_width_curve: {metrics['lower_face_width_curve']:.3f}")
    print()
    print(f"Форма лица: {result['label']}")
    print(f"shape_gate={result['shape_gate']}")
    print(f"Второй кандидат: {result['secondary_candidate']} (margin={result['margin']:.3f})")
    if result["low_confidence"]:
        print("Предупреждение: низкая уверенность, форма близка к смешанному типу.")
    print("Обоснование:", ", ".join(result["evidence"]))
    print("Confidence:")
    for k, v in sorted(confidence.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
