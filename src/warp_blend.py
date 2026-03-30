from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _compute_canvas(base_image: np.ndarray, incoming_image: np.ndarray, homography: np.ndarray):
    base_h, base_w = base_image.shape[:2]
    in_h, in_w = incoming_image.shape[:2]

    base_corners = np.float32([[0, 0], [base_w, 0], [base_w, base_h], [0, base_h]]).reshape(-1, 1, 2)
    incoming_corners = np.float32([[0, 0], [in_w, 0], [in_w, in_h], [0, in_h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(incoming_corners, homography)
    all_corners = np.concatenate((base_corners, warped_corners), axis=0)

    [x_min, y_min] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [x_max, y_max] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translate_x = -x_min if x_min < 0 else 0
    translate_y = -y_min if y_min < 0 else 0

    translation = np.array(
        [
            [1.0, 0.0, float(translate_x)],
            [0.0, 1.0, float(translate_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    width = int(x_max - x_min)
    height = int(y_max - y_min)
    return translation, width, height


def _soft_mask(mask: np.ndarray, blur_kernel: int) -> np.ndarray:
    kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    soft = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
    return soft[..., None]


def crop_valid_region(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(binary)
    if points is None:
        return image
    x, y, width, height = cv2.boundingRect(points)
    return image[y : y + height, x : x + width]


def stitch_pair(base_image: np.ndarray, incoming_image: np.ndarray, homography: np.ndarray, config: dict):
    translation, width, height = _compute_canvas(base_image, incoming_image, homography)
    total_transform = translation @ homography

    warped_incoming = cv2.warpPerspective(incoming_image, total_transform, (width, height))
    warped_mask = cv2.warpPerspective(
        np.full(incoming_image.shape[:2], 255, dtype=np.uint8),
        total_transform,
        (width, height),
    )

    base_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    base_mask = np.zeros((height, width), dtype=np.uint8)

    tx = int(translation[0, 2])
    ty = int(translation[1, 2])
    base_h, base_w = base_image.shape[:2]
    base_canvas[ty : ty + base_h, tx : tx + base_w] = base_image
    base_mask[ty : ty + base_h, tx : tx + base_w] = 255

    blend_cfg = config["blend"]
    mode = str(blend_cfg.get("mode", "feather")).lower()

    if mode == "average":
        overlap = (base_mask > 0) & (warped_mask > 0)
        only_base = (base_mask > 0) & ~overlap
        only_incoming = (warped_mask > 0) & ~overlap
        result = np.zeros_like(base_canvas)
        result[only_base] = base_canvas[only_base]
        result[only_incoming] = warped_incoming[only_incoming]
        result[overlap] = ((base_canvas[overlap].astype(np.float32) + warped_incoming[overlap].astype(np.float32)) / 2.0).astype(
            np.uint8
        )
    else:
        blur_kernel = int(blend_cfg.get("feather_blur_kernel", 61))
        base_weight = _soft_mask(base_mask / 255.0, blur_kernel)
        incoming_weight = _soft_mask(warped_mask / 255.0, blur_kernel)
        total_weight = base_weight + incoming_weight
        total_weight[total_weight == 0] = 1.0
        blended = (
            base_canvas.astype(np.float32) * base_weight + warped_incoming.astype(np.float32) * incoming_weight
        ) / total_weight
        result = np.clip(blended, 0, 255).astype(np.uint8)

    cropped = crop_valid_region(result)
    debug = {
        "canvas_size": {"width": width, "height": height},
        "translation": {"x": float(translation[0, 2]), "y": float(translation[1, 2])},
    }
    return cropped, debug
