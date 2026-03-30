from __future__ import annotations

from typing import List, Tuple

import cv2

from .utils import orb_score_type


def create_feature_extractor(config: dict):
    feature_cfg = config["feature"]
    method = feature_cfg["name"].lower()
    if method != "orb":
        raise ValueError(f"Unsupported feature extractor: {method}")

    orb_cfg = feature_cfg["orb"]
    return cv2.ORB_create(
        nfeatures=int(orb_cfg["nfeatures"]),
        scaleFactor=float(orb_cfg["scaleFactor"]),
        nlevels=int(orb_cfg["nlevels"]),
        edgeThreshold=int(orb_cfg["edgeThreshold"]),
        firstLevel=int(orb_cfg["firstLevel"]),
        WTA_K=int(orb_cfg["WTA_K"]),
        scoreType=orb_score_type(str(orb_cfg["scoreType"])),
        patchSize=int(orb_cfg["patchSize"]),
        fastThreshold=int(orb_cfg["fastThreshold"]),
    )


def extract_features(extractor, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray | None]:
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors
