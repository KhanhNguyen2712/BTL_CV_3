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


def match_descriptors(matcher, desc1, desc2, config: dict) -> Tuple[List[list], List[cv2.DMatch]]:
    if desc1 is None or desc2 is None:
        return [], []

    knn_k = int(config["matching"].get("knn_k", 2))
    raw_matches = matcher.knnMatch(desc1, desc2, k=knn_k)
    ratio = float(config["matching"].get("ratio", 0.75))

    good_matches: List[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good_matches.append(first)

    return raw_matches, good_matches
