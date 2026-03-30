from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .features import create_feature_extractor, extract_features
from .homography import estimate_homography
from .matching import build_matcher, match_descriptors
from .preprocess import ImageData, preprocess_image
from .utils import write_image
from .warp_blend import stitch_pair


@dataclass
class PanoramaState:
    name: str
    color: np.ndarray
    gray: np.ndarray


def _draw_keypoints(image: np.ndarray, keypoints) -> np.ndarray:
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


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
        raw_preview = cv2.hconcat([base_gray, incoming_gray])

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
        good_preview = cv2.hconcat([base_gray, incoming_gray])

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
        inlier_preview = cv2.hconcat([base_gray, incoming_gray])
    return raw_preview, good_preview, inlier_preview


def _save_debug_images(output_dir: Path, step_name: str, state: PanoramaState, image: ImageData, debug_cfg: dict, base_kp, incoming_kp, raw_matches, good_matches, mask):
    if debug_cfg.get("save_keypoints", True):
        write_image(output_dir / f"{state.name}_keypoints.jpg", _draw_keypoints(state.color, base_kp))
        write_image(output_dir / f"{image.name}_keypoints.jpg", _draw_keypoints(image.color, incoming_kp))

    if debug_cfg.get("save_matches", True):
        raw_preview, good_preview, inlier_preview = _draw_matches(
            state.gray,
            base_kp,
            image.gray,
            incoming_kp,
            raw_matches,
            good_matches,
            mask,
        )
        write_image(output_dir / f"{step_name}_matches_raw.jpg", raw_preview)
        write_image(output_dir / f"{step_name}_matches_good.jpg", good_preview)
        write_image(output_dir / f"{step_name}_matches_inliers.jpg", inlier_preview)


def _prepare_reference(images: List[ImageData]) -> tuple[PanoramaState, List[ImageData], List[ImageData], int]:
    reference_idx = len(images) // 2
    reference = images[reference_idx]
    left = list(reversed(images[:reference_idx]))
    right = images[reference_idx + 1 :]
    state = PanoramaState(name=reference.name, color=reference.color.copy(), gray=reference.gray.copy())
    return state, left, right, reference_idx


def _interleave_from_center(left_images: List[ImageData], right_images: List[ImageData]) -> List[ImageData]:
    ordered: List[ImageData] = []
    max_len = max(len(left_images), len(right_images))
    for idx in range(max_len):
        if idx < len(left_images):
            ordered.append(left_images[idx])
        if idx < len(right_images):
            ordered.append(right_images[idx])
    return ordered


def run_panorama(images: List[ImageData], config: dict, output_dir: str | Path) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extractor = create_feature_extractor(config)
    matcher = build_matcher(config)
    debug_cfg = config["debug"]
    runtime_cfg = config["runtime"]

    state, left_images, right_images, reference_idx = _prepare_reference(images)
    summary = {
        "feature": config["feature"]["name"],
        "reference_index": reference_idx,
        "reference_name": state.name,
        "images": [img.name for img in images],
        "steps": [],
        "status": "running",
    }

    ordered_images = _interleave_from_center(left_images, right_images)
    for step_index, image in enumerate(ordered_images, start=1):
        step_name = f"step_{step_index:02d}_{image.name}"

        base_kp, base_desc = extract_features(extractor, state.gray)
        incoming_kp, incoming_desc = extract_features(extractor, image.gray)

        raw_matches, good_matches = match_descriptors(matcher, base_desc, incoming_desc, config)
        homography, mask, homography_stats = estimate_homography(base_kp, incoming_kp, good_matches, config)

        _save_debug_images(
            output_path,
            step_name,
            state,
            image,
            debug_cfg,
            base_kp,
            incoming_kp,
            raw_matches,
            good_matches,
            mask,
        )

        step_summary = {
            "step": step_index,
            "base_name": state.name,
            "incoming_name": image.name,
            "base_keypoints": len(base_kp),
            "incoming_keypoints": len(incoming_kp),
            "raw_match_groups": len(raw_matches),
            "good_matches": len(good_matches),
            "homography": homography_stats,
        }

        if homography is None or not homography_stats["success"]:
            step_summary["status"] = "failed"
            summary["steps"].append(step_summary)
            if runtime_cfg.get("stop_on_failure", True):
                summary["status"] = "failed"
                summary["failure_reason"] = f"Failed at {step_name}: {homography_stats['reason']}"
                summary["failed_step"] = step_index
                summary["final_panorama_shape"] = list(state.color.shape)
                write_image(output_path / "panorama_final.jpg", state.color)
                return summary
            continue

        # Matches were computed from panorama -> incoming. We need incoming -> panorama to warp the new image onto the current canvas.
        try:
            incoming_to_base = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            step_summary["status"] = "failed"
            step_summary["homography"]["reason"] = "Homography matrix is singular and cannot be inverted."
            step_summary["homography"]["success"] = False
            summary["steps"].append(step_summary)
            summary["status"] = "failed"
            summary["failure_reason"] = f"Failed at {step_name}: homography matrix is singular."
            summary["failed_step"] = step_index
            summary["final_panorama_shape"] = list(state.color.shape)
            write_image(output_path / "panorama_final.jpg", state.color)
            return summary

        panorama, warp_debug = stitch_pair(state.color, image.color, incoming_to_base, config)
        state = PanoramaState(
            name=f"{state.name}__{image.name}",
            color=panorama,
            gray=preprocess_image(panorama, config["preprocess"]),
        )

        if debug_cfg.get("save_intermediate_panorama", True):
            write_image(output_path / f"{step_name}_panorama.jpg", panorama)

        step_summary["status"] = "ok"
        step_summary["warp"] = warp_debug
        step_summary["panorama_shape"] = list(panorama.shape)
        summary["steps"].append(step_summary)

    summary["status"] = "ok"
    summary["final_panorama_shape"] = list(state.color.shape)
    write_image(output_path / "panorama_final.jpg", state.color)
    return summary
