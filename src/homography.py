from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def _normalize_homography(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix.astype(np.float64)
    scale = normalized[2, 2]
    if abs(scale) > 1e-8:
        normalized = normalized / scale
    return normalized


def _affine_to_homography(matrix: np.ndarray) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = matrix.astype(np.float64)
    return homography


def _failure_stats(matches_count: int, reason: str) -> Dict:
    return {
        "good_matches": matches_count,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "success": False,
        "reason": reason,
        "canvas_bounds": {},
        "model": "homography",
    }


def _canvas_bounds(base_shape: tuple[int, ...], incoming_shape: tuple[int, ...], incoming_to_base: np.ndarray) -> Dict[str, int | Dict[str, float]]:
    base_h, base_w = base_shape[:2]
    incoming_h, incoming_w = incoming_shape[:2]

    base_corners = np.float32([[0, 0], [base_w, 0], [base_w, base_h], [0, base_h]]).reshape(-1, 1, 2)
    incoming_corners = np.float32([[0, 0], [incoming_w, 0], [incoming_w, incoming_h], [0, incoming_h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(incoming_corners, incoming_to_base)
    all_corners = np.concatenate((base_corners, warped_corners), axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    incoming_center = np.array([incoming_w / 2.0, incoming_h / 2.0], dtype=np.float32).reshape(1, 1, 2)
    base_center = np.array([base_w / 2.0, base_h / 2.0], dtype=np.float32).reshape(1, 1, 2)
    projected_center = cv2.perspectiveTransform(incoming_center, incoming_to_base).reshape(2)
    base_center = base_center.reshape(2)

    return {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
        "width": int(x_max - x_min),
        "height": int(y_max - y_min),
        "pixels": int((x_max - x_min) * (y_max - y_min)),
        "center_shift": {
            "dx": float(projected_center[0] - base_center[0]),
            "dy": float(projected_center[1] - base_center[1]),
        },
    }


def _pairwise_limits(config: dict, model: str) -> tuple[int, int, int, float, float, float]:
    homography_cfg = config["homography"]
    max_width = int(homography_cfg.get("pairwise_max_canvas_width", 4000))
    max_height = int(homography_cfg.get("pairwise_max_canvas_height", 2500))
    max_pixels = int(homography_cfg.get("pairwise_max_canvas_pixels", 8000000))
    max_vertical_shift = float(homography_cfg.get("pairwise_max_vertical_shift", 500))
    max_shift_ratio_x = float(homography_cfg.get("pairwise_max_center_shift_ratio_x", 0.6))
    max_shift_ratio_y = float(homography_cfg.get("pairwise_max_center_shift_ratio_y", 0.2))

    # Affine fallback is allowed to shift a bit more horizontally because it is now
    # the preferred fallback for the provided right-to-left panorama sequence.
    if model == "affine_partial":
        max_shift_ratio_x = max(max_shift_ratio_x, 0.75)

    return max_width, max_height, max_pixels, max_vertical_shift, max_shift_ratio_x, max_shift_ratio_y


def _global_limits(config: dict) -> tuple[int, int, int]:
    homography_cfg = config["homography"]
    return (
        int(homography_cfg.get("max_canvas_width", 12000)),
        int(homography_cfg.get("max_canvas_height", 6000)),
        int(homography_cfg.get("max_canvas_pixels", 30000000)),
    )


def _validate_transform(
    base_shape: tuple[int, ...],
    incoming_shape: tuple[int, ...],
    incoming_to_base: np.ndarray,
    config: dict,
    *,
    pairwise: bool,
    model: str,
) -> Tuple[bool, str, Dict]:
    if not np.isfinite(incoming_to_base).all():
        return False, "Transform contains NaN or Inf values.", {}

    normalized = _normalize_homography(incoming_to_base)
    bounds = _canvas_bounds(base_shape, incoming_shape, normalized)

    if bounds["width"] <= 0 or bounds["height"] <= 0:
        return False, "Projected canvas has non-positive dimensions.", bounds

    if pairwise:
        max_width, max_height, max_pixels, max_vertical_shift, max_shift_ratio_x, max_shift_ratio_y = _pairwise_limits(
            config, model
        )
    else:
        max_width, max_height, max_pixels = _global_limits(config)
        max_vertical_shift = max_shift_ratio_x = max_shift_ratio_y = 0.0

    if bounds["width"] > max_width:
        return False, f"Projected canvas width {bounds['width']} exceeds limit {max_width}.", bounds
    if bounds["height"] > max_height:
        return False, f"Projected canvas height {bounds['height']} exceeds limit {max_height}.", bounds
    if bounds["pixels"] > max_pixels:
        return False, f"Projected canvas area {bounds['pixels']} exceeds limit {max_pixels}.", bounds

    if not pairwise:
        return True, "ok", bounds

    base_h, base_w = base_shape[:2]
    delta_x = abs(bounds["center_shift"]["dx"])
    delta_y = abs(bounds["center_shift"]["dy"])

    if delta_y > max_vertical_shift:
        return False, f"Projected vertical shift {delta_y:.2f} exceeds limit {max_vertical_shift}.", bounds
    if delta_x > base_w * max_shift_ratio_x:
        return False, f"Projected horizontal center shift {delta_x:.2f} exceeds ratio limit {max_shift_ratio_x}.", bounds
    if delta_y > base_h * max_shift_ratio_y:
        return False, f"Projected vertical center shift {delta_y:.2f} exceeds ratio limit {max_shift_ratio_y}.", bounds
    if delta_y > delta_x and delta_x > 1.0:
        return False, "Projected motion is dominated by vertical shift, inconsistent with horizontal panorama.", bounds

    capture_direction = str(config["homography"].get("capture_direction", "")).lower()
    signed_dx = bounds["center_shift"]["dx"]
    if capture_direction == "right_to_left" and signed_dx <= 0:
        return False, "Projected horizontal shift is not positive for a right-to-left capture sequence.", bounds
    if capture_direction == "left_to_right" and signed_dx >= 0:
        return False, "Projected horizontal shift is not negative for a left-to-right capture sequence.", bounds

    return True, "ok", bounds


def _build_model_stats(matches_count: int, inliers: int, model_matrix: np.ndarray, canvas_bounds: Dict, success: bool, reason: str, model: str) -> Dict:
    return {
        "good_matches": matches_count,
        "inliers": inliers,
        "inlier_ratio": inliers / max(matches_count, 1),
        "success": success,
        "reason": reason,
        "canvas_bounds": canvas_bounds,
        "matrix": model_matrix.round(6).tolist(),
        "model": model,
    }


def _evaluate_model(
    matrix: np.ndarray | None,
    mask: np.ndarray | None,
    *,
    base_shape: tuple[int, ...],
    incoming_shape: tuple[int, ...],
    matches_count: int,
    min_inliers: int,
    min_inlier_ratio: float,
    config: dict,
    pairwise: bool,
    model: str,
) -> tuple[np.ndarray | None, np.ndarray | None, Dict | None]:
    if matrix is None or mask is None:
        return None, None, None

    normalized = _normalize_homography(matrix if model == "homography" else _affine_to_homography(matrix))
    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(matches_count, 1)
    is_valid, reason, canvas_bounds = _validate_transform(
        base_shape,
        incoming_shape,
        normalized,
        config,
        pairwise=pairwise,
        model=model,
    )
    success = inliers >= min_inliers and inlier_ratio >= min_inlier_ratio and is_valid

    if success:
        reason = "ok"
    elif is_valid:
        reason = f"{model} rejected by inlier thresholds."

    return normalized, mask, _build_model_stats(matches_count, inliers, normalized, canvas_bounds, success, reason, model)


def estimate_homography(
    kp_src,
    kp_dst,
    matches: List[cv2.DMatch],
    base_shape: tuple[int, ...],
    incoming_shape: tuple[int, ...],
    config: dict,
    pairwise: bool = False,
) -> Tuple[np.ndarray | None, np.ndarray | None, Dict]:
    min_good_matches = int(config["matching"].get("min_good_matches", 20))
    if len(matches) < min_good_matches:
        return None, None, _failure_stats(
            len(matches),
            f"Only {len(matches)} good matches, below configured minimum {min_good_matches}.",
        )
    if len(matches) < 4:
        return None, None, _failure_stats(len(matches), "Not enough matches to estimate homography.")

    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography_cfg = config["homography"]
    min_inliers = int(homography_cfg["min_inliers"])
    min_inlier_ratio = float(homography_cfg["min_inlier_ratio"])
    affine_min_inliers = int(homography_cfg.get("affine_min_inliers", min_inliers))
    affine_min_inlier_ratio = float(homography_cfg.get("affine_min_inlier_ratio", min_inlier_ratio))
    ransac_threshold = float(homography_cfg["ransac_thresh"])

    homography_matrix, homography_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    homography_matrix, homography_mask, homography_stats = _evaluate_model(
        homography_matrix,
        homography_mask,
        base_shape=base_shape,
        incoming_shape=incoming_shape,
        matches_count=len(matches),
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        config=config,
        pairwise=pairwise,
        model="homography",
    )
    if homography_stats and homography_stats["success"]:
        return homography_matrix, homography_mask, homography_stats

    if pairwise and homography_cfg.get("fallback_affine_partial", True):
        affine_matrix, affine_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
        )
        affine_matrix, affine_mask, affine_stats = _evaluate_model(
            affine_matrix,
            affine_mask,
            base_shape=base_shape,
            incoming_shape=incoming_shape,
            matches_count=len(matches),
            min_inliers=affine_min_inliers,
            min_inlier_ratio=affine_min_inlier_ratio,
            config=config,
            pairwise=pairwise,
            model="affine_partial",
        )
        if affine_stats and affine_stats["success"]:
            return affine_matrix, affine_mask, affine_stats
        if affine_stats:
            if homography_stats is not None:
                affine_stats["fallback_from"] = homography_stats
                affine_stats["reason"] = f"Homography failed: {homography_stats['reason']}; affine_partial failed: {affine_stats['reason']}"
            return None, affine_mask, affine_stats

    if homography_stats is not None:
        return None, homography_mask, homography_stats

    return None, None, _failure_stats(len(matches), "RANSAC could not estimate a valid homography.")
