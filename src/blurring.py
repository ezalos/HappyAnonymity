# ABOUTME: Face blurring utilities for video anonymization
# ABOUTME: Applies Gaussian blur to bounding box regions in video frames

from __future__ import annotations

import numpy as np
import cv2


def blur_faces(
    frame: np.ndarray,
    bboxes: list[np.ndarray],
    kernel_size: int = 99,
) -> np.ndarray:
    """Apply Gaussian blur to face regions in a frame.

    Args:
        frame: BGR image (H, W, 3)
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        kernel_size: Gaussian blur kernel size (must be odd)

    Returns:
        Frame with faces blurred.
    """
    result = frame.copy()
    h, w = frame.shape[:2]

    if kernel_size % 2 == 0:
        kernel_size += 1

    for bbox in bboxes:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = result[y1:y2, x1:x2]
        result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 30)

    return result
