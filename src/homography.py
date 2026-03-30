from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def estimate_homography(kp_src, kp_dst, matches: List[cv2.DMatch], config: dict) -> Tuple[np.ndarray | None, np.ndarray | None, Dict]:
    min_good_matches = int(config["matching"].get("min_good_matches", 20))
    if len(matches) < min_good_matches:
        return None, None, {
            "good_matches": len(matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "success": False,
            "reason": f"Only {len(matches)} good matches, below configured minimum {min_good_matches}.",
        }

    if len(matches) < 4:
        return None, None, {
            "good_matches": len(matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "success": False,
            "reason": "Not enough matches to estimate homography.",
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
        }

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(len(matches), 1)

    success = (
        inliers >= int(homography_cfg["min_inliers"])
        and inlier_ratio >= float(homography_cfg["min_inlier_ratio"])
        and np.isfinite(matrix).all()
    )

    reason = "ok" if success else "Homography rejected by inlier thresholds."
    stats = {
        "good_matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "success": success,
        "reason": reason,
    }
    return matrix, mask, stats
