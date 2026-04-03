from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from .utils import orb_score_type


@dataclass
class FeatureExtractorBundle:
    name: str
    detector: object
    detection_width: int | None = None


def _resize_for_detection(image: np.ndarray, detection_width: int | None) -> tuple[np.ndarray, float]:
    if detection_width is None or detection_width <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    if width <= detection_width:
        return image, 1.0

    scale = detection_width / float(width)
    resized = cv2.resize(
        image,
        (detection_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def create_feature_extractor(config: dict):
    feature_cfg = config["feature"]
    method = feature_cfg["name"].lower()
    if method == "orb":
        orb_cfg = feature_cfg["orb"]
        detector = cv2.ORB_create(
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
        return FeatureExtractorBundle(name="orb", detector=detector, detection_width=None)

    if method == "sift":
        sift_cfg = feature_cfg["sift"]
        detector = cv2.SIFT_create(
            nfeatures=int(sift_cfg.get("nfeatures", 500)),
            nOctaveLayers=int(sift_cfg.get("nOctaveLayers", 3)),
            contrastThreshold=float(sift_cfg.get("contrastThreshold", 0.04)),
            edgeThreshold=float(sift_cfg.get("edgeThreshold", 10)),
            sigma=float(sift_cfg.get("sigma", 1.6)),
        )
        detection_width = sift_cfg.get("detection_width")
        detection_width = int(detection_width) if detection_width is not None else None
        return FeatureExtractorBundle(name="sift", detector=detector, detection_width=detection_width)

    raise ValueError(f"Unsupported feature extractor: {method}")


def extract_features(extractor: FeatureExtractorBundle, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray | None]:
    detection_image, scale = _resize_for_detection(image, extractor.detection_width)
    keypoints, descriptors = extractor.detector.detectAndCompute(detection_image, None)
    if scale < 1.0:
        for keypoint in keypoints:
            keypoint.pt = (keypoint.pt[0] / scale, keypoint.pt[1] / scale)
            keypoint.size /= scale
    return keypoints, descriptors
