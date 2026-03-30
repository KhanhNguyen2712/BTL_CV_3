from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class ImageData:
    name: str
    path: str
    color: np.ndarray
    gray: np.ndarray
    scale: float
    original_shape: tuple[int, int, int]


def resize_keep_aspect(image: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return image.copy(), 1.0

    scale = max_long_edge / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_image(image: np.ndarray, preprocess_cfg: dict) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if preprocess_cfg.get("gaussian_blur", True):
        kernel_size = int(preprocess_cfg.get("gaussian_kernel", 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    if preprocess_cfg.get("equalize_hist", False):
        gray = cv2.equalizeHist(gray)

    return gray


def load_images(input_dir: str | Path, config: dict) -> List[ImageData]:
    image_dir = Path(input_dir)
    image_paths = sorted([path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if len(image_paths) < 2:
        raise ValueError("Need at least two input images to build a panorama.")

    resize_cfg = config["resize"]
    preprocess_cfg = config["preprocess"]
    images: List[ImageData] = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")

        resized, scale = resize_keep_aspect(image, int(resize_cfg["max_long_edge"]))
        gray = preprocess_image(resized, preprocess_cfg)
        images.append(
            ImageData(
                name=path.stem,
                path=str(path),
                color=resized,
                gray=gray,
                scale=scale,
                original_shape=image.shape,
            )
        )

    return images
