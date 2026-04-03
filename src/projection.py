from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .preprocess import ImageData, preprocess_image


@dataclass
class ProjectedImageData:
    name: str
    color: np.ndarray
    gray: np.ndarray
    focal_length: float


def estimate_focal_length(image: np.ndarray, fov_degrees: float) -> float:
    height, width = image.shape[:2]
    del height
    return (width / 2.0) / np.tan(np.radians(fov_degrees / 2.0))


def cylindrical_project(image: np.ndarray, focal_length: float) -> np.ndarray:
    height, width = image.shape[:2]
    x_center = np.arange(width, dtype=np.float32) - width / 2.0
    y_center = np.arange(height, dtype=np.float32) - height / 2.0
    grid_x, grid_y = np.meshgrid(x_center, y_center)
    theta = grid_x / focal_length
    cylindrical_y = grid_y / np.sqrt(grid_x**2 + focal_length**2)
    map_x = (focal_length * np.tan(theta) + width / 2.0).astype(np.float32)
    map_y = (cylindrical_y * focal_length / np.cos(theta) + height / 2.0).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def project_images_cylindrical(images: List[ImageData], config: dict) -> List[ProjectedImageData]:
    preprocess_cfg = config["preprocess"]
    cylindrical_cfg = config["projection"]["cylindrical"]
    fov_degrees = float(cylindrical_cfg.get("fov_degrees", 60.0))

    projected: List[ProjectedImageData] = []
    for image in images:
        focal_length = estimate_focal_length(image.color, fov_degrees)
        projected_color = cylindrical_project(image.color, focal_length)
        projected_gray = preprocess_image(projected_color, preprocess_cfg)
        projected.append(
            ProjectedImageData(
                name=image.name,
                color=projected_color,
                gray=projected_gray,
                focal_length=float(focal_length),
            )
        )
    return projected


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, float(dx)],
            [0.0, 1.0, float(dy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
