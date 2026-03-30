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


def _validate_homography(
    base_shape: tuple[int, ...],
    incoming_shape: tuple[int, ...],
    incoming_to_base: np.ndarray,
    config: dict,
    pairwise: bool = False,
    model: str = "homography",
) -> Tuple[bool, str, Dict]:
    if not np.isfinite(incoming_to_base).all():
        return False, "Homography contains NaN or Inf values.", {}

    normalized = _normalize_homography(incoming_to_base)

    bounds = _canvas_bounds(base_shape, incoming_shape, normalized)
    base_h, base_w = base_shape[:2]
    incoming_h, incoming_w = incoming_shape[:2]
    base_center = np.array([base_w / 2.0, base_h / 2.0], dtype=np.float32).reshape(1, 1, 2)
    incoming_center = np.array([incoming_w / 2.0, incoming_h / 2.0], dtype=np.float32).reshape(1, 1, 2)
    projected_center = cv2.perspectiveTransform(incoming_center, normalized).reshape(2)
    base_center = base_center.reshape(2)
    delta_x = float(projected_center[0] - base_center[0])
    delta_y = float(projected_center[1] - base_center[1])
    bounds["center_shift"] = {"dx": delta_x, "dy": delta_y}

    homography_cfg = config["homography"]
    if pairwise:
        max_width = int(homography_cfg.get("pairwise_max_canvas_width", 4000))
        max_height = int(homography_cfg.get("pairwise_max_canvas_height", 2500))
        max_pixels = int(homography_cfg.get("pairwise_max_canvas_pixels", 8000000))
    else:
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

    if pairwise:
        max_vertical_shift = float(homography_cfg.get("pairwise_max_vertical_shift", 500))
        if abs(delta_y) > max_vertical_shift:
            return False, f"Projected vertical shift {delta_y:.2f} exceeds limit {max_vertical_shift}.", bounds

        max_center_shift_ratio_x = float(homography_cfg.get("pairwise_max_center_shift_ratio_x", 0.45))
        max_center_shift_ratio_y = float(homography_cfg.get("pairwise_max_center_shift_ratio_y", 0.2))
        if abs(delta_x) > base_w * max_center_shift_ratio_x:
            return False, f"Projected horizontal center shift {delta_x:.2f} exceeds ratio limit {max_center_shift_ratio_x}.", bounds

        if abs(delta_y) > base_h * max_center_shift_ratio_y:
            return False, f"Projected vertical center shift {delta_y:.2f} exceeds ratio limit {max_center_shift_ratio_y}.", bounds

        if abs(delta_y) > abs(delta_x) and abs(delta_x) > 1.0:
            return False, "Projected motion is dominated by vertical shift, inconsistent with horizontal panorama.", bounds

        capture_direction = str(homography_cfg.get("capture_direction", "")).lower()
        if capture_direction == "right_to_left" and delta_x <= 0:
            return False, "Projected horizontal shift is not positive for a right-to-left capture sequence.", bounds
        if capture_direction == "left_to_right" and delta_x >= 0:
            return False, "Projected horizontal shift is not negative for a left-to-right capture sequence.", bounds

    return True, "ok", bounds


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
    min_inliers = int(homography_cfg["min_inliers"])
    min_inlier_ratio = float(homography_cfg["min_inlier_ratio"])
    affine_min_inliers = int(homography_cfg.get("affine_min_inliers", min_inliers))
    affine_min_inlier_ratio = float(homography_cfg.get("affine_min_inlier_ratio", min_inlier_ratio))

    homography_matrix, homography_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, float(homography_cfg["ransac_thresh"])
    )

    homography_stats = None
    if homography_matrix is not None and homography_mask is not None:
        homography_matrix = _normalize_homography(homography_matrix)
        homography_inliers = int(homography_mask.ravel().sum())
        homography_inlier_ratio = homography_inliers / max(len(matches), 1)
        valid_homography, homography_reason, homography_canvas_bounds = _validate_homography(
            base_shape,
            incoming_shape,
            homography_matrix,
            config,
            pairwise=pairwise,
            model="homography",
        )
        homography_success = (
            homography_inliers >= min_inliers
            and homography_inlier_ratio >= min_inlier_ratio
            and valid_homography
        )
        if homography_success:
            homography_reason = "ok"
        elif valid_homography:
            homography_reason = "Homography rejected by inlier thresholds."

        homography_stats = {
            "good_matches": len(matches),
            "inliers": homography_inliers,
            "inlier_ratio": homography_inlier_ratio,
            "success": homography_success,
            "reason": homography_reason,
            "canvas_bounds": homography_canvas_bounds,
            "matrix": homography_matrix.round(6).tolist(),
            "model": "homography",
        }
        if homography_success:
            return homography_matrix, homography_mask, homography_stats

    if pairwise and homography_cfg.get("fallback_affine_partial", True):
        affine_matrix, affine_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(homography_cfg["ransac_thresh"]),
        )
        if affine_matrix is not None and affine_mask is not None:
            affine_h = _affine_to_homography(affine_matrix)
            affine_h = _normalize_homography(affine_h)
            affine_inliers = int(affine_mask.ravel().sum())
            affine_inlier_ratio = affine_inliers / max(len(matches), 1)
            valid_affine, affine_reason, affine_canvas_bounds = _validate_homography(
                base_shape,
                incoming_shape,
                affine_h,
                config,
                pairwise=pairwise,
                model="affine_partial",
            )
            affine_success = (
                affine_inliers >= affine_min_inliers
                and affine_inlier_ratio >= affine_min_inlier_ratio
                and valid_affine
            )
            if affine_success:
                affine_reason = "ok"
            elif valid_affine:
                affine_reason = "Affine partial rejected by inlier thresholds."

            affine_stats = {
                "good_matches": len(matches),
                "inliers": affine_inliers,
                "inlier_ratio": affine_inlier_ratio,
                "success": affine_success,
                "reason": affine_reason,
                "canvas_bounds": affine_canvas_bounds,
                "matrix": affine_h.round(6).tolist(),
                "model": "affine_partial",
            }
            if affine_success:
                return affine_h, affine_mask, affine_stats

            failure_reason = affine_reason
            if homography_stats is not None:
                failure_reason = f"Homography failed: {homography_stats['reason']}; affine_partial failed: {affine_reason}"
            affine_stats["reason"] = failure_reason
            affine_stats["fallback_from"] = homography_stats
            return None, affine_mask, affine_stats

    if homography_stats is not None:
        return None, homography_mask, homography_stats

    return None, None, {
        "good_matches": len(matches),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "success": False,
        "reason": "RANSAC could not estimate a valid homography.",
        "canvas_bounds": {},
        "model": "homography",
    }
