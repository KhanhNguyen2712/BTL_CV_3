from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_image(path: str | Path, image) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image to {output_path}")


def orb_score_type(name: str) -> int:
    mapping = {
        "HARRIS_SCORE": cv2.ORB_HARRIS_SCORE,
        "FAST_SCORE": cv2.ORB_FAST_SCORE,
    }
    return mapping.get(name, cv2.ORB_HARRIS_SCORE)
