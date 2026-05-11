import cv2
import numpy as np


def _laplacian_sharpness(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, v / 500.0)


def _exposure_score(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    m = float(np.mean(gray)) / 255.0
    d = abs(m - 0.45)
    return max(0.0, 1.0 - d * 2.5)


def assess_photo(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None,
    min_sharpness: float = 0.08,
    min_face_coverage: float = 0.04,
    min_exposure: float = 0.25,
) -> tuple[dict, bool, list[str]]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharp = _laplacian_sharpness(gray)
    exp = _exposure_score(gray)
    issues: list[str] = []
    if face_bbox is None:
        coverage = 0.0
        issues.append("face_not_detected")
    else:
        x, y, bw, bh = face_bbox
        coverage = (bw * bh) / max(1, w * h)
        if coverage < min_face_coverage:
            issues.append("face_too_small")
    passes = (
        sharp >= min_sharpness
        and exp >= min_exposure
        and (face_bbox is not None and coverage >= min_face_coverage)
    )
    if sharp < min_sharpness:
        issues.append("blurry")
    if exp < min_exposure:
        issues.append("under_or_overexposed")
    return (
        {
            "sharpness_score": round(sharp, 4),
            "exposure_score": round(exp, 4),
            "face_coverage_ratio": round(coverage, 4),
        },
        passes,
        issues,
    )
