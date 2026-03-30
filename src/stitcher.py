from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .features import create_feature_extractor, extract_features
from .homography import estimate_homography
from .matching import build_matcher, match_descriptors
from .preprocess import ImageData
from .utils import write_image
from .warp_blend import compose_panorama


def _draw_keypoints(image: np.ndarray, keypoints) -> np.ndarray:
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def _concat_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left.shape) == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if len(right.shape) == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    if left.shape[0] != right.shape[0]:
        target_height = max(left.shape[0], right.shape[0])
        left = cv2.copyMakeBorder(left, 0, target_height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        right = cv2.copyMakeBorder(right, 0, target_height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return cv2.hconcat([left, right])


def _draw_matches(base_gray, base_kp, incoming_gray, incoming_kp, raw_matches, good_matches, mask):
    if raw_matches:
        raw_preview = cv2.drawMatchesKnn(
            base_gray,
            base_kp,
            incoming_gray,
            incoming_kp,
            raw_matches[:100],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        raw_preview = _concat_preview(base_gray, incoming_gray)

    if good_matches:
        good_preview = cv2.drawMatches(
            base_gray,
            base_kp,
            incoming_gray,
            incoming_kp,
            good_matches[:100],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        good_preview = _concat_preview(base_gray, incoming_gray)

    inlier_matches = []
    if mask is not None:
        for match, keep in zip(good_matches, mask.ravel().tolist()):
            if keep:
                inlier_matches.append(match)

    if inlier_matches:
        inlier_preview = cv2.drawMatches(
            base_gray,
            base_kp,
            incoming_gray,
            incoming_kp,
            inlier_matches[:100],
            None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        inlier_preview = _concat_preview(base_gray, incoming_gray)
    return raw_preview, good_preview, inlier_preview


def _save_pair_debug_images(
    output_dir: Path,
    step_name: str,
    left_image: ImageData,
    right_image: ImageData,
    debug_cfg: dict,
    left_kp,
    right_kp,
    raw_matches,
    good_matches,
    mask,
) -> None:
    if debug_cfg.get("save_keypoints", True):
        write_image(output_dir / f"{left_image.name}_keypoints.jpg", _draw_keypoints(left_image.color, left_kp))
        write_image(output_dir / f"{right_image.name}_keypoints.jpg", _draw_keypoints(right_image.color, right_kp))

    if debug_cfg.get("save_matches", True):
        raw_preview, good_preview, inlier_preview = _draw_matches(
            left_image.gray,
            left_kp,
            right_image.gray,
            right_kp,
            raw_matches,
            good_matches,
            mask,
        )
        write_image(output_dir / f"{step_name}_matches_raw.jpg", raw_preview)
        write_image(output_dir / f"{step_name}_matches_good.jpg", good_preview)
        write_image(output_dir / f"{step_name}_matches_inliers.jpg", inlier_preview)


def _estimate_pairwise_homographies(images: List[ImageData], config: dict, output_path: Path) -> tuple[List[np.ndarray], List[Dict]]:
    extractor = create_feature_extractor(config)
    matcher = build_matcher(config)
    debug_cfg = config["debug"]

    pairwise_homographies: List[np.ndarray] = []
    step_summaries: List[Dict] = []

    for idx in range(len(images) - 1):
        left_image = images[idx]
        right_image = images[idx + 1]
        step_name = f"pair_{idx:02d}_{left_image.name}_to_{right_image.name}"

        left_kp, left_desc = extract_features(extractor, left_image.gray)
        right_kp, right_desc = extract_features(extractor, right_image.gray)
        raw_matches, good_matches = match_descriptors(matcher, left_desc, right_desc, config)
        homography, mask, homography_stats = estimate_homography(
            left_kp,
            right_kp,
            good_matches,
            right_image.color.shape,
            left_image.color.shape,
            config,
            pairwise=True,
        )

        _save_pair_debug_images(
            output_path,
            step_name,
            left_image,
            right_image,
            debug_cfg,
            left_kp,
            right_kp,
            raw_matches,
            good_matches,
            mask,
        )

        step_summary = {
            "step": idx + 1,
            "left_name": left_image.name,
            "right_name": right_image.name,
            "left_keypoints": len(left_kp),
            "right_keypoints": len(right_kp),
            "raw_match_groups": len(raw_matches),
            "good_matches": len(good_matches),
            "homography": homography_stats,
        }

        if homography is None or not homography_stats["success"]:
            step_summary["status"] = "failed"
            step_summaries.append(step_summary)
            raise RuntimeError(f"Failed at {step_name}: {homography_stats['reason']}")

        step_summary["status"] = "ok"
        pairwise_homographies.append(homography)
        step_summaries.append(step_summary)

    return pairwise_homographies, step_summaries


def _accumulate_transforms(images: List[ImageData], pairwise_homographies: List[np.ndarray], reference_idx: int) -> List[np.ndarray]:
    transforms: List[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in images]
    transforms[reference_idx] = np.eye(3, dtype=np.float64)

    for idx in range(reference_idx - 1, -1, -1):
        transforms[idx] = transforms[idx + 1] @ pairwise_homographies[idx]
        scale = transforms[idx][2, 2]
        if abs(scale) > 1e-8:
            transforms[idx] = transforms[idx] / scale

    for idx in range(reference_idx + 1, len(images)):
        try:
            inverse_h = np.linalg.inv(pairwise_homographies[idx - 1])
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Pairwise homography between images {idx - 1} and {idx} is singular.") from exc
        transforms[idx] = transforms[idx - 1] @ inverse_h
        scale = transforms[idx][2, 2]
        if abs(scale) > 1e-8:
            transforms[idx] = transforms[idx] / scale

    return transforms


def run_panorama(images: List[ImageData], config: dict, output_dir: str | Path) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reference_idx = len(images) // 2

    summary = {
        "feature": config["feature"]["name"],
        "reference_index": reference_idx,
        "reference_name": images[reference_idx].name,
        "images": [img.name for img in images],
        "steps": [],
        "status": "running",
    }

    try:
        pairwise_homographies, step_summaries = _estimate_pairwise_homographies(images, config, output_path)
        summary["steps"].extend(step_summaries)
        transforms = _accumulate_transforms(images, pairwise_homographies, reference_idx)
        summary["global_transforms"] = {
            image.name: transforms[idx].round(6).tolist() for idx, image in enumerate(images)
        }
        panorama, composition_debug = compose_panorama(images, transforms, config)
    except RuntimeError as exc:
        summary["status"] = "failed"
        summary["failure_reason"] = str(exc)
        summary["failed_stage"] = "pairwise_matching"
        return summary
    except ValueError as exc:
        summary["status"] = "failed"
        summary["failure_reason"] = str(exc)
        summary["failed_stage"] = "global_composition"
        summary["global_transforms"] = summary.get("global_transforms", {})
        return summary

    write_image(output_path / "panorama_final.jpg", panorama)

    if config["debug"].get("save_intermediate_panorama", True):
        write_image(output_path / "panorama_composed.jpg", panorama)

    summary["composition"] = composition_debug
    summary["final_panorama_shape"] = list(panorama.shape)
    summary["status"] = "ok"
    return summary
