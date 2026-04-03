from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def build_matcher(config: dict):
    matching_cfg = config["matching"]
    matcher_name = str(matching_cfg.get("matcher", "bf")).lower()
    norm_name = str(matching_cfg.get("norm", "hamming")).lower()
    norm = cv2.NORM_HAMMING if norm_name == "hamming" else cv2.NORM_L2
    if matcher_name == "bf":
        return {
            "name": "bf",
            "matcher": cv2.BFMatcher(norm, crossCheck=False),
            "norm": norm,
            "requires_float32": False,
        }

    if matcher_name == "flann":
        if norm != cv2.NORM_L2:
            raise ValueError("FLANN matcher is only supported with L2 descriptors.")
        index_params = {"algorithm": 1, "trees": int(matching_cfg.get("flann_trees", 5))}
        search_params = {"checks": int(matching_cfg.get("flann_checks", 50))}
        return {
            "name": "flann",
            "matcher": cv2.FlannBasedMatcher(index_params, search_params),
            "norm": norm,
            "requires_float32": True,
        }

    raise ValueError(f"Unsupported matcher: {matcher_name}")


def _ratio_filter(raw_matches: List[list], ratio: float) -> List[cv2.DMatch]:
    filtered: List[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            filtered.append(first)
    return filtered


def _symmetry_filter(forward_matches: List[cv2.DMatch], backward_matches: List[cv2.DMatch]) -> List[cv2.DMatch]:
    backward_pairs = {(match.trainIdx, match.queryIdx) for match in backward_matches}
    return [match for match in forward_matches if (match.queryIdx, match.trainIdx) in backward_pairs]


def _grid_occupancy(matches: List[cv2.DMatch], keypoints, image_shape: tuple[int, ...], rows: int, cols: int) -> int:
    if not matches:
        return 0

    height, width = image_shape[:2]
    occupied = set()
    for match in matches:
        x, y = keypoints[match.queryIdx].pt
        col = min(cols - 1, int((x / max(width, 1)) * cols))
        row = min(rows - 1, int((y / max(height, 1)) * rows))
        occupied.add((row, col))
    return len(occupied)


def _cross_check_matches(desc1, desc2, norm: int) -> List[cv2.DMatch]:
    cross_matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = cross_matcher.match(desc1, desc2)
    return sorted(matches, key=lambda match: match.distance)


def _prepare_descriptors(desc, requires_float32: bool):
    if desc is None:
        return None
    if requires_float32 and desc.dtype != np.float32:
        return desc.astype(np.float32)
    return desc


def _limit_matches(matches: List[cv2.DMatch], max_good_matches: int | None) -> List[cv2.DMatch]:
    if max_good_matches is None or max_good_matches <= 0:
        return sorted(matches, key=lambda match: match.distance)
    return sorted(matches, key=lambda match: match.distance)[:max_good_matches]


def match_descriptors(matcher_bundle, desc1, desc2, kp1, image1_shape, config: dict) -> Tuple[List[list], List[cv2.DMatch], Dict]:
    if desc1 is None or desc2 is None:
        return [], [], {
            "strategy": "none",
            "grid_occupancy": 0,
        }

    matching_cfg = config["matching"]
    matcher = matcher_bundle["matcher"]
    requires_float32 = bool(matcher_bundle.get("requires_float32", False))
    prepared_desc1 = _prepare_descriptors(desc1, requires_float32)
    prepared_desc2 = _prepare_descriptors(desc2, requires_float32)
    knn_k = int(matching_cfg.get("knn_k", 2))
    ratio = float(matching_cfg.get("ratio", 0.75))
    max_good_matches = matching_cfg.get("max_good_matches")
    max_good_matches = int(max_good_matches) if max_good_matches is not None else None
    raw_matches = matcher.knnMatch(prepared_desc1, prepared_desc2, k=knn_k)
    forward_good_matches = _ratio_filter(raw_matches, ratio)

    if not matching_cfg.get("symmetry_check", True):
        filtered_matches = forward_good_matches
        strategy = "knn_ratio"
    else:
        backward_raw_matches = matcher.knnMatch(prepared_desc2, prepared_desc1, k=knn_k)
        backward_good_matches = _ratio_filter(backward_raw_matches, ratio)
        filtered_matches = _symmetry_filter(forward_good_matches, backward_good_matches)
        strategy = "knn_ratio_symmetry"

    filtered_matches = _limit_matches(filtered_matches, max_good_matches)

    rows = int(matching_cfg.get("pairwise_grid_rows", 2))
    cols = int(matching_cfg.get("pairwise_grid_cols", 3))
    min_cells = int(matching_cfg.get("pairwise_min_occupied_cells", 3))
    occupancy = _grid_occupancy(filtered_matches, kp1, image1_shape, rows, cols)

    if occupancy < min_cells and matching_cfg.get("fallback_cross_check", True):
        filtered_matches = _cross_check_matches(prepared_desc1, prepared_desc2, matcher_bundle["norm"])
        filtered_matches = _limit_matches(filtered_matches, max_good_matches)
        strategy = "bf_crosscheck_fallback"
        occupancy = _grid_occupancy(filtered_matches, kp1, image1_shape, rows, cols)

    metadata = {
        "strategy": strategy,
        "matcher": matcher_bundle["name"],
        "grid_occupancy": occupancy,
        "grid_rows": rows,
        "grid_cols": cols,
        "min_required_cells": min_cells,
        "max_good_matches": max_good_matches,
    }
    return raw_matches, filtered_matches, metadata
