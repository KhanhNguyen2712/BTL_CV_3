from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageData:
    name: str
    path: str
    color: np.ndarray
    gray: np.ndarray
    scale: float
    original_shape: tuple[int, int, int]


@dataclass(frozen=True)
class InputSequence:
    name: str
    path: str


def _list_image_paths(image_dir: Path) -> List[Path]:
    return sorted([path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS])


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


def discover_input_sequences(input_dir: str | Path) -> List[InputSequence]:
    root_dir = Path(input_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {root_dir}")

    direct_images = _list_image_paths(root_dir)
    if direct_images:
        return [InputSequence(name=root_dir.name, path=str(root_dir))]

    sequences: List[InputSequence] = []
    for child_dir in sorted([path for path in root_dir.iterdir() if path.is_dir()]):
        if _list_image_paths(child_dir):
            sequences.append(InputSequence(name=child_dir.name, path=str(child_dir)))

    if not sequences:
        raise ValueError(f"No supported images found in {root_dir} or its first-level subdirectories.")

    return sequences


def load_images(input_dir: str | Path, config: dict) -> List[ImageData]:
    image_dir = Path(input_dir)
    image_paths = _list_image_paths(image_dir)
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
