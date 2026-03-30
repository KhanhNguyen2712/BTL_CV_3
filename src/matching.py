from __future__ import annotations

from typing import List, Tuple

import cv2


def build_matcher(config: dict):
    matching_cfg = config["matching"]
    matcher_name = str(matching_cfg.get("matcher", "bf")).lower()
    norm_name = str(matching_cfg.get("norm", "hamming")).lower()

    if matcher_name != "bf":
        raise ValueError(f"Unsupported matcher: {matcher_name}")

    norm = cv2.NORM_HAMMING if norm_name == "hamming" else cv2.NORM_L2
    return cv2.BFMatcher(norm, crossCheck=False)


def _ratio_filter(raw_matches: List[list], ratio: float) -> List[cv2.DMatch]:
    filtered: List[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            filtered.append(first)
    return filtered


def match_descriptors(matcher, desc1, desc2, config: dict) -> Tuple[List[list], List[cv2.DMatch]]:
    if desc1 is None or desc2 is None:
        return [], []

    knn_k = int(config["matching"].get("knn_k", 2))
    ratio = float(config["matching"].get("ratio", 0.75))
    raw_matches = matcher.knnMatch(desc1, desc2, k=knn_k)
    forward_good_matches = _ratio_filter(raw_matches, ratio)

    if not config["matching"].get("symmetry_check", True):
        return raw_matches, forward_good_matches

    backward_raw_matches = matcher.knnMatch(desc2, desc1, k=knn_k)
    backward_good_matches = _ratio_filter(backward_raw_matches, ratio)
    backward_pairs = {(match.trainIdx, match.queryIdx) for match in backward_good_matches}

    symmetric_matches = [
        match for match in forward_good_matches if (match.queryIdx, match.trainIdx) in backward_pairs
    ]
    return raw_matches, symmetric_matches
