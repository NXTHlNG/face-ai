import os
import cv2
import json
import urllib.request
import numpy as np
import mediapipe as mp

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

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

# Верхний ряд по овальному контуру (старая линия лба + верх щёк)
OLD_FOREHEAD_SEQ = [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251]

# Нижний ряд под ним (более низкие точки по лбу)
LOWER_FOREHEAD_SEQ = [71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def to_px(lm, w, h):
    return np.array([
        int(np.clip(lm.x * w, 0, w - 1)),
        int(np.clip(lm.y * h, 0, h - 1))
    ], dtype=np.float32)


def get_pts(landmarks, ids, w, h):
    return [to_px(landmarks[i], w, h) for i in ids]


def bezier_arc(p0, p1, p2, n=30):
    pts = []
    for t in np.linspace(0, 1, n):
        pt = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
        pts.append(pt)
    return pts

def refine_temple_on_boundary(gray, center_pt, temple_hint, max_shift, samples=45):
    c = np.array(center_pt, dtype=np.float32)
    hint = np.array(temple_hint, dtype=np.float32)
    vec = hint - c
    dist = np.linalg.norm(vec)
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
    shift_len = np.linalg.norm(shift)
    if shift_len > max_shift:
        best_pt = hint + shift * (max_shift / (shift_len + 1e-6))
    return best_pt


def estimate_forehead(image, face_oval, left_brow, right_brow, landmarks, w, h):
    chin = max(face_oval, key=lambda p: p[1])
    left_temple_hint = min(get_pts(landmarks, LEFT_TEMPLE_HINTS, w, h), key=lambda p: p[0])
    right_temple_hint = max(get_pts(landmarks, RIGHT_TEMPLE_HINTS, w, h), key=lambda p: p[0])

    brow_center = np.mean(np.array(left_brow + right_brow), axis=0)
    temple_mid = (left_temple_hint + right_temple_hint) / 2.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    temple_width_hint = np.linalg.norm(right_temple_hint - left_temple_hint)
    max_shift = max(6.0, temple_width_hint * 0.09)

    left_temple = refine_temple_on_boundary(
        gray=gray,
        center_pt=temple_mid,
        temple_hint=left_temple_hint,
        max_shift=max_shift
    )
    right_temple = refine_temple_on_boundary(
        gray=gray,
        center_pt=temple_mid,
        temple_hint=right_temple_hint,
        max_shift=max_shift
    )

    face_height = chin[1] - temple_mid[1]
    forehead_raise = face_height * 0.32

    forehead_top = np.array([
        temple_mid[0],
        max(0, temple_mid[1] - forehead_raise)
    ], dtype=np.float32)

    left_forehead = left_temple.astype(np.float32)
    right_forehead = right_temple.astype(np.float32)

    forehead_arc = bezier_arc(left_forehead, forehead_top, right_forehead, n=25)

    return {
        "left_temple": left_forehead,
        "right_temple": right_forehead,
        "forehead_top": forehead_top,
        "forehead_arc": forehead_arc,
        "chin": chin
    }


def build_face_oval_with_upper_row(face_oval_pts, landmarks, w, h, forehead_top):
    """
    1) берём OLD и LOWER из landmarks;
    2) считаем UP1 = 2*OLD - LOWER (двойной подъём относительно нижнего ряда);
    3) поднимаем весь UP1 по y так, чтобы точка 10 совпала с forehead_top;
    4) вставляем итоговый ряд вместо OLD в FACE_OVAL.
    """
    pts = np.array(face_oval_pts, dtype=np.float32)

    # --- шаг 1: OLD и LOW в пикселях по парам ---
    if len(OLD_FOREHEAD_SEQ) != len(LOWER_FOREHEAD_SEQ):
        raise ValueError("OLD_FOREHEAD_SEQ и LOWER_FOREHEAD_SEQ должны быть одинаковой длины")

    old_pts = []
    low_pts = []
    for old_idx, low_idx in zip(OLD_FOREHEAD_SEQ, LOWER_FOREHEAD_SEQ):
        old_pts.append(to_px(landmarks[old_idx], w, h))
        low_pts.append(to_px(landmarks[low_idx], w, h))
    old_pts = np.array(old_pts, dtype=np.float32)
    low_pts = np.array(low_pts, dtype=np.float32)

    # --- шаг 2: двойной подъём относительно LOWER: up1 = 2*OLD - LOW ---
    up1_pts = 2 * old_pts - low_pts

    # --- шаг 3: дополнительный подъём, чтобы точка 10 попала в forehead_top ---
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

    # ---------- СУЖАЕМ ШЕСТЬ ВЕРХНИХ ТОЧЕК ЛОБА ПО ГОРИЗОНТАЛИ ----------
    squeeze_k = 0.05  # твой коэффициент

    min_x = np.min(up2_pts[:, 0])
    max_x = np.max(up2_pts[:, 0])
    center_x = (min_x + max_x) / 2.0

    width = max_x - min_x
    shift0 = width * squeeze_k        # крайние точки
    shift1 = shift0                   # следующие
    shift2 = shift0 * 0.5             # ещё ближе к центру

    n = len(up2_pts)
    left_idx0 = 0
    right_idx0 = n - 1
    left_idx1 = 1
    right_idx1 = n - 2
    left_idx2 = 2
    right_idx2 = n - 3

    # 1) самые крайние над висками
    up2_pts[left_idx0, 0] = min(center_x, up2_pts[left_idx0, 0] + shift0)
    up2_pts[right_idx0, 0] = max(center_x, up2_pts[right_idx0, 0] - shift0)

    # 2) следующие к центру
    up2_pts[left_idx1, 0] = min(center_x, up2_pts[left_idx1, 0] + shift1)
    up2_pts[right_idx1, 0] = max(center_x, up2_pts[right_idx1, 0] - shift1)

    # 3) ещё две точки ближе к центру
    up2_pts[left_idx2, 0] = min(center_x, up2_pts[left_idx2, 0] + shift2)
    up2_pts[right_idx2, 0] = max(center_x, up2_pts[right_idx2, 0] - shift2)
    # -------------------------------------------------------------------

    # --- шаг 4: вставляем up2_pts вместо OLD_FOREHEAD_SEQ в FACE_OVAL ---
    result_pts = []
    old_set = set(OLD_FOREHEAD_SEQ)
    inserted = False

    for idx in FACE_OVAL:
        if idx in OLD_FOREHEAD_SEQ:
            if not inserted:
                for p in up2_pts:
                    result_pts.append(p.tolist())
                inserted = True
            continue
        else:
            result_pts.append(pts[FACE_OVAL.index(idx)].tolist())

    return np.array(result_pts, dtype=np.float32)


def build_ordered_face_contour(face_oval_pts):
    pts = np.array(face_oval_pts, dtype=np.float32)

    center_x = np.mean(pts[:, 0])
    chin_idx = np.argmax(pts[:, 1])

    left_side = pts[pts[:, 0] <= center_x]
    right_side = pts[pts[:, 0] > center_x]

    left_side = sorted(left_side.tolist(), key=lambda p: p[1])      # сверху вниз
    right_side = sorted(right_side.tolist(), key=lambda p: p[1])    # сверху вниз

    chin = pts[chin_idx].tolist()

    if left_side[-1] != chin:
        left_side.append(chin)
    if right_side[-1] != chin:
        right_side.append(chin)

    contour = []
    contour.extend(left_side)
    contour.extend(right_side[::-1][1:])

    return np.array(contour, dtype=np.int32)


def draw_poly(img, pts, color, closed=False, thickness=2):
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [arr], closed, color, thickness, cv2.LINE_AA)


def draw_pts(img, pts, color, r=2):
    for p in pts:
        cv2.circle(img, tuple(np.int32(p)), r, color, -1)


def analyze_face(
    image_path,
    output_path="D:/PXL_annotated_fixed_combined_f_0.jpg",
    json_path="D:/face_points_fixed_combined.json"
):
    ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Не удалось открыть изображение.")

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("Лицо не найдено.")

    landmarks = result.face_landmarks[0]
    face_oval = get_pts(landmarks, FACE_OVAL, w, h)
    left_brow = get_pts(landmarks, LEFT_BROW, w, h)
    right_brow = get_pts(landmarks, RIGHT_BROW, w, h)

    forehead = estimate_forehead(image, face_oval, left_brow, right_brow, landmarks, w, h)

    face_oval_modified = build_face_oval_with_upper_row(
        face_oval_pts=face_oval,
        landmarks=landmarks,
        w=w,
        h=h,
        forehead_top=forehead["forehead_top"]
    )

    full_contour = build_ordered_face_contour(face_oval_modified)

    out = image.copy()

    draw_poly(out, full_contour, (0, 255, 0), closed=True, thickness=2)
    draw_pts(out, full_contour, (0, 255, 0), r=2)

    cv2.circle(out, tuple(np.int32(forehead["left_temple"])), 5, (0, 0, 255), -1)
    cv2.circle(out, tuple(np.int32(forehead["right_temple"])), 5, (0, 0, 255), -1)
    cv2.circle(out, tuple(np.int32(forehead["forehead_top"])), 6, (255, 255, 0), -1)

    cv2.imwrite(output_path, out)

    # ------ ЗАПИСЬ ТОЧЕК КОНТУРА В JSON ------
    data = {
        "image_size": {"width": w, "height": h},
        "full_face_contour": full_contour.astype(int).tolist(),
        "face_oval_modified": np.array(face_oval_modified).astype(int).tolist(),
        "forehead_top": np.array(forehead["forehead_top"]).astype(int).tolist(),
        "chin": np.array(forehead["chin"]).astype(int).tolist()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # -----------------------------------------

    return data


if __name__ == "__main__":
    analyze_face("D:/tr.png")