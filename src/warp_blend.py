from __future__ import annotations

import cv2
import numpy as np


def _global_canvas_bounds(images, transforms):
    all_corners = []
    for image, transform in zip(images, transforms):
        height, width = image.color.shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, transform))

    merged = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.floor(merged.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(merged.max(axis=0).ravel()).astype(int)

    translation = np.array(
        [
            [1.0, 0.0, float(-x_min if x_min < 0 else 0)],
            [0.0, 1.0, float(-y_min if y_min < 0 else 0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return translation, int(x_max - x_min), int(y_max - y_min)


def _feather_weight(mask: np.ndarray, blur_kernel: int) -> np.ndarray:
    binary_mask = (mask > 0).astype(np.uint8)
    if binary_mask.max() == 0:
        return np.zeros(mask.shape + (1,), dtype=np.float32)

    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    distance[binary_mask == 0] = 0.0
    if distance.max() > 0:
        distance = distance / distance.max()

    kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    weight = cv2.GaussianBlur(distance, (kernel, kernel), 0)
    weight[binary_mask == 0] = 0.0
    return weight[..., None].astype(np.float32)


def crop_valid_region(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    binary = (valid_mask > 0).astype(np.uint8) * 255
    points = cv2.findNonZero(binary)
    if points is None:
        return image
    x, y, width, height = cv2.boundingRect(points)
    return image[y : y + height, x : x + width]


def compose_panorama(images, transforms, config: dict):
    translation, width, height = _global_canvas_bounds(images, transforms)
    homography_cfg = config["homography"]
    max_width = int(homography_cfg.get("max_canvas_width", 12000))
    max_height = int(homography_cfg.get("max_canvas_height", 6000))
    max_pixels = int(homography_cfg.get("max_canvas_pixels", 30000000))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid global canvas size: width={width}, height={height}")
    if width > max_width or height > max_height or width * height > max_pixels:
        raise ValueError(
            "Global panorama canvas exceeds configured safety limits: "
            f"width={width}, height={height}, pixels={width * height}"
        )

    accum_image = np.zeros((height, width, 3), dtype=np.float32)
    accum_weight = np.zeros((height, width, 1), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=np.uint8)
    blur_kernel = int(config["blend"].get("feather_blur_kernel", 61))

    image_debug = []
    for image, transform in zip(images, transforms):
        total_transform = translation @ transform
        warped_image = cv2.warpPerspective(image.color, total_transform, (width, height))
        warped_mask = cv2.warpPerspective(
            np.full(image.color.shape[:2], 255, dtype=np.uint8),
            total_transform,
            (width, height),
        )

        weight = _feather_weight(warped_mask, blur_kernel)
        accum_image += warped_image.astype(np.float32) * weight
        accum_weight += weight
        valid_mask = np.maximum(valid_mask, warped_mask)
        image_debug.append({"name": image.name, "shape": list(image.color.shape)})

    accum_weight[accum_weight == 0] = 1.0
    panorama_full = np.clip(accum_image / accum_weight, 0, 255).astype(np.uint8)
    panorama_full[valid_mask == 0] = 0
    panorama_cropped = crop_valid_region(panorama_full, valid_mask)

    return panorama_full, panorama_cropped, {
        "canvas_size": {"width": width, "height": height},
        "translation": {"x": float(translation[0, 2]), "y": float(translation[1, 2])},
        "images": image_debug,
        "valid_pixels": int((valid_mask > 0).sum()),
    }
