from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def _canvas_bounds(base_shape: tuple[int, ...], incoming_shape: tuple[int, ...], incoming_to_base: np.ndarray) -> Dict[str, int]:
    base_h, base_w = base_shape[:2]
    incoming_h, incoming_w = incoming_shape[:2]

    base_corners = np.float32([[0, 0], [base_w, 0], [base_w, base_h], [0, base_h]]).reshape(-1, 1, 2)
    incoming_corners = np.float32([[0, 0], [incoming_w, 0], [incoming_w, incoming_h], [0, incoming_h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(incoming_corners, incoming_to_base)
    all_corners = np.concatenate((base_corners, warped_corners), axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    pixels = width * height
    return {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
        "width": width,
        "height": height,
        "pixels": pixels,
    }


def _validate_homography(base_shape: tuple[int, ...], incoming_shape: tuple[int, ...], homography: np.ndarray, config: dict) -> Tuple[bool, str, Dict]:
    if not np.isfinite(homography).all():
        return False, "Homography contains NaN or Inf values.", {}

    try:
        incoming_to_base = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return False, "Homography matrix is singular and cannot be inverted.", {}

    bounds = _canvas_bounds(base_shape, incoming_shape, incoming_to_base)
    homography_cfg = config["homography"]
    max_width = int(homography_cfg.get("max_canvas_width", 12000))
    max_height = int(homography_cfg.get("max_canvas_height", 6000))
    max_pixels = int(homography_cfg.get("max_canvas_pixels", 30000000))

    if bounds["width"] <= 0 or bounds["height"] <= 0:
        return False, "Projected canvas has non-positive dimensions.", bounds

    if bounds["width"] > max_width:
        return False, f"Projected canvas width {bounds['width']} exceeds limit {max_width}.", bounds

    if bounds["height"] > max_height:
        return False, f"Projected canvas height {bounds['height']} exceeds limit {max_height}.", bounds

    if bounds["pixels"] > max_pixels:
        return False, f"Projected canvas area {bounds['pixels']} exceeds limit {max_pixels}.", bounds

    return True, "ok", bounds


def estimate_homography(
    kp_src,
    kp_dst,
    matches: List[cv2.DMatch],
    base_shape: tuple[int, ...],
    incoming_shape: tuple[int, ...],
    config: dict,
) -> Tuple[np.ndarray | None, np.ndarray | None, Dict]:
    min_good_matches = int(config["matching"].get("min_good_matches", 20))
    if len(matches) < min_good_matches:
        return None, None, {
            "good_matches": len(matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "success": False,
            "reason": f"Only {len(matches)} good matches, below configured minimum {min_good_matches}.",
            "canvas_bounds": {},
        }

    if len(matches) < 4:
        return None, None, {
            "good_matches": len(matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "success": False,
            "reason": "Not enough matches to estimate homography.",
            "canvas_bounds": {},
        }

    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography_cfg = config["homography"]
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(homography_cfg["ransac_thresh"]))
    if matrix is None or mask is None:
        return None, None, {
            "good_matches": len(matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "success": False,
            "reason": "RANSAC could not estimate a valid homography.",
            "canvas_bounds": {},
        }

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(len(matches), 1)
    valid_homography, reason, canvas_bounds = _validate_homography(base_shape, incoming_shape, matrix, config)

    success = (
        inliers >= int(homography_cfg["min_inliers"])
        and inlier_ratio >= float(homography_cfg["min_inlier_ratio"])
        and valid_homography
    )

    if success:
        reason = "ok"
    elif valid_homography:
        reason = "Homography rejected by inlier thresholds."

    stats = {
        "good_matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "success": success,
        "reason": reason,
        "canvas_bounds": canvas_bounds,
    }
    return matrix, mask, stats
