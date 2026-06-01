from __future__ import annotations

import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image_bgr
