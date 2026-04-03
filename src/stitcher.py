from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .features import create_feature_extractor, extract_features
from .homography import estimate_homography
from .matching import build_matcher, match_descriptors
from .preprocess import ImageData
from .projection import ProjectedImageData, project_images_cylindrical, translation_matrix
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
        raw_matches, good_matches, match_metadata = match_descriptors(
            matcher,
            left_desc,
            right_desc,
            left_kp,
            left_image.color.shape,
            config,
        )
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
            "matching": match_metadata,
            "homography": homography_stats,
        }

        if homography is None or not homography_stats["success"]:
            step_summary["status"] = "failed"
            step_summaries.append(step_summary)
            raise PairwiseEstimationError(
                f"Failed at {step_name}: {homography_stats['reason']}",
                step_summaries,
            )

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


def _estimate_cylindrical_translation(kp_src, kp_dst, good_matches, config: dict):
    cylindrical_cfg = config["projection"]["cylindrical"]
    min_good_matches = int(cylindrical_cfg.get("min_good_matches", 10))
    if len(good_matches) < min_good_matches:
        return None, None, {
            "success": False,
            "reason": f"Only {len(good_matches)} good matches, below configured minimum {min_good_matches}.",
            "model": "translation_from_homography",
            "matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        float(cylindrical_cfg.get("ransac_thresh", 5.0)),
    )
    if homography is None or mask is None:
        return None, None, {
            "success": False,
            "reason": "RANSAC could not estimate a valid homography.",
            "model": "translation_from_homography",
            "matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(len(good_matches), 1)
    dx = float(homography[0, 2])
    dy = float(homography[1, 2])
    return translation_matrix(dx, dy), mask, {
        "success": True,
        "reason": "ok",
        "model": "translation_from_homography",
        "matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "translation": {"dx": dx, "dy": dy},
        "homography_matrix": homography.round(6).tolist(),
    }


def _estimate_pairwise_cylindrical_transforms(
    projected_images: List[ProjectedImageData],
    config: dict,
    output_path: Path,
) -> tuple[dict[int, np.ndarray | None], List[Dict], Dict[str, Dict]]:
    extractor = create_feature_extractor(config)
    matcher = build_matcher(config)
    debug_cfg = config["debug"]
    step_summaries: List[Dict] = []
    pairwise_transforms: dict[int, np.ndarray | None] = {}
    pairwise_translations: Dict[str, Dict] = {}

    if debug_cfg.get("save_keypoints", True):
        for image in projected_images:
            write_image(output_path / f"{image.name}_cylindrical.jpg", image.color)

    for idx in range(1, len(projected_images)):
        source_image = projected_images[idx]
        target_image = projected_images[idx - 1]
        step_name = f"pair_{idx - 1:02d}_{source_image.name}_to_{target_image.name}_cylindrical"

        source_kp, source_desc = extract_features(extractor, source_image.gray)
        target_kp, target_desc = extract_features(extractor, target_image.gray)
        raw_matches, good_matches, match_metadata = match_descriptors(
            matcher,
            source_desc,
            target_desc,
            source_kp,
            source_image.color.shape,
            config,
        )
        transform, mask, translation_stats = _estimate_cylindrical_translation(
            source_kp,
            target_kp,
            good_matches,
            config,
        )

        _save_pair_debug_images(
            output_path,
            step_name,
            source_image,
            target_image,
            debug_cfg,
            source_kp,
            target_kp,
            raw_matches,
            good_matches,
            mask,
        )

        step_summary = {
            "step": idx,
            "source_name": source_image.name,
            "target_name": target_image.name,
            "left_name": target_image.name,
            "right_name": source_image.name,
            "source_keypoints": len(source_kp),
            "target_keypoints": len(target_kp),
            "raw_match_groups": len(raw_matches),
            "good_matches": len(good_matches),
            "matching": match_metadata,
            "homography": translation_stats,
            "status": "ok" if transform is not None else "failed",
        }
        step_summaries.append(step_summary)

        pair_key = f"{source_image.name}_to_{target_image.name}"
        if transform is None:
            pairwise_transforms[idx] = None
            pairwise_translations[pair_key] = {
                "success": False,
                "reason": translation_stats["reason"],
            }
            continue

        pairwise_transforms[idx] = transform
        pairwise_translations[pair_key] = {
            "success": True,
            "dx": float(transform[0, 2]),
            "dy": float(transform[1, 2]),
            "inliers": translation_stats["inliers"],
            "inlier_ratio": translation_stats["inlier_ratio"],
        }

    return pairwise_transforms, step_summaries, pairwise_translations


def _accumulate_cylindrical_transforms(
    projected_images: List[ProjectedImageData],
    pairwise_transforms: dict[int, np.ndarray | None],
    reference_idx: int,
) -> tuple[dict[int, np.ndarray], List[int]]:
    transforms: dict[int, np.ndarray] = {reference_idx: np.eye(3, dtype=np.float64)}

    for idx in range(reference_idx + 1, len(projected_images)):
        pair_transform = pairwise_transforms.get(idx)
        if pair_transform is None:
            break
        transforms[idx] = transforms[idx - 1] @ pair_transform

    for idx in range(reference_idx - 1, -1, -1):
        pair_transform = pairwise_transforms.get(idx + 1)
        if pair_transform is None:
            break
        transforms[idx] = transforms[idx + 1] @ np.linalg.inv(pair_transform)

    return transforms, sorted(transforms.keys())


def _feather_blend_onto(canvas: np.ndarray, warped: np.ndarray) -> np.ndarray:
    canvas_mask = (canvas.sum(axis=2) > 0).astype(np.uint8)
    warped_mask = (warped.sum(axis=2) > 0).astype(np.uint8)

    only_canvas = (canvas_mask > 0) & (warped_mask == 0)
    only_warped = (warped_mask > 0) & (canvas_mask == 0)
    overlap = (canvas_mask > 0) & (warped_mask > 0)

    distance_canvas = cv2.distanceTransform(canvas_mask, cv2.DIST_L2, 5).astype(np.float32)
    distance_warped = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 5).astype(np.float32)
    total_distance = distance_canvas + distance_warped + 1e-6

    result = np.zeros_like(canvas)
    result[only_canvas] = canvas[only_canvas]
    result[only_warped] = warped[only_warped]

    weight_canvas = (distance_canvas / total_distance)[..., None]
    weight_warped = (distance_warped / total_distance)[..., None]
    blended = (
        weight_canvas * canvas.astype(np.float32) +
        weight_warped * warped.astype(np.float32)
    ).astype(np.uint8)
    result[overlap] = blended[overlap]
    return result


def _crop_black_border(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(threshold)
    if points is None:
        return image
    x, y, width, height = cv2.boundingRect(points)
    return image[y : y + height, x : x + width]


def _compose_cylindrical_panorama(
    projected_images: List[ProjectedImageData],
    transforms: dict[int, np.ndarray],
    reference_idx: int,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    valid_indices = sorted(transforms.keys())
    all_corners = []
    for idx in valid_indices:
        image = projected_images[idx]
        height, width = image.color.shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, transforms[idx]))

    merged = np.concatenate(all_corners, axis=0)
    x_min = int(np.floor(merged[:, :, 0].min())) - 1
    y_min = int(np.floor(merged[:, :, 1].min())) - 1
    x_max = int(np.ceil(merged[:, :, 0].max())) + 1
    y_max = int(np.ceil(merged[:, :, 1].max())) + 1
    width = x_max - x_min
    height = y_max - y_min

    cylindrical_cfg = config["projection"]["cylindrical"]
    max_canvas_dim = int(cylindrical_cfg.get("max_canvas_dim", 12000))
    canvas_scale = 1.0
    if max(width, height) > max_canvas_dim:
        canvas_scale = min(max_canvas_dim / max(width, 1), max_canvas_dim / max(height, 1))
        width = max(1, int(round(width * canvas_scale)))
        height = max(1, int(round(height * canvas_scale)))

    translation = np.array(
        [
            [canvas_scale, 0.0, float(-x_min * canvas_scale)],
            [0.0, canvas_scale, float(-y_min * canvas_scale)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    blend_order = [reference_idx] + list(range(reference_idx + 1, len(projected_images))) + list(range(reference_idx - 1, -1, -1))
    applied_order: List[int] = []
    for idx in blend_order:
        if idx not in transforms:
            continue
        warped = cv2.warpPerspective(projected_images[idx].color, translation @ transforms[idx], (width, height))
        canvas = _feather_blend_onto(canvas, warped)
        applied_order.append(idx)

    cropped = _crop_black_border(canvas)
    return canvas, cropped, {
        "canvas_size": {"width": width, "height": height},
        "canvas_scale": canvas_scale,
        "translation": {"x": float(translation[0, 2]), "y": float(translation[1, 2])},
        "valid_image_indices": valid_indices,
        "blend_order": applied_order,
    }


class PairwiseEstimationError(RuntimeError):
    def __init__(self, message: str, step_summaries: List[Dict]):
        super().__init__(message)
        self.step_summaries = step_summaries


def _build_summary(images: List[ImageData], reference_idx: int, feature_name: str, projection_mode: str) -> Dict:
    return {
        "feature": feature_name,
        "projection_mode": projection_mode,
        "reference_index": reference_idx,
        "reference_name": images[reference_idx].name,
        "images": [img.name for img in images],
        "steps": [],
        "status": "running",
    }


def _run_planar_panorama(images: List[ImageData], config: dict, output_path: Path) -> Dict:
    reference_idx = len(images) // 2
    summary = _build_summary(images, reference_idx, config["feature"]["name"], "planar")

    try:
        pairwise_homographies, step_summaries = _estimate_pairwise_homographies(images, config, output_path)
        summary["steps"].extend(step_summaries)
        transforms = _accumulate_transforms(images, pairwise_homographies, reference_idx)
        summary["global_transforms"] = {
            image.name: transforms[idx].round(6).tolist() for idx, image in enumerate(images)
        }
        panorama_full, panorama_cropped, composition_debug = compose_panorama(images, transforms, config)
    except PairwiseEstimationError as exc:
        summary["steps"].extend(exc.step_summaries)
        summary["status"] = "failed"
        summary["failure_reason"] = str(exc)
        summary["failed_stage"] = "pairwise_matching"
        return summary
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

    write_image(output_path / "panorama_final.jpg", panorama_cropped)
    summary["composition"] = composition_debug
    summary["panorama_full_shape"] = list(panorama_full.shape)
    summary["final_panorama_shape"] = list(panorama_cropped.shape)
    summary["status"] = "ok"
    return summary


def _run_cylindrical_panorama(images: List[ImageData], config: dict, output_path: Path) -> Dict:
    if str(config["feature"]["name"]).lower() != "sift":
        raise ValueError("Projection mode 'cylindrical' is only supported with feature extractor 'sift'.")

    reference_idx = len(images) // 2
    summary = _build_summary(images, reference_idx, config["feature"]["name"], "cylindrical")
    cylindrical_cfg = config["projection"]["cylindrical"]
    motion_model = str(cylindrical_cfg.get("motion_model", "translation_from_homography")).lower()
    blend_order_mode = str(cylindrical_cfg.get("blend_order", "center_out")).lower()
    if motion_model != "translation_from_homography":
        raise ValueError(f"Unsupported cylindrical motion model: {motion_model}")
    if blend_order_mode != "center_out":
        raise ValueError(f"Unsupported cylindrical blend order: {blend_order_mode}")
    summary["motion_model"] = motion_model
    summary["blend_order_mode"] = blend_order_mode

    projected_images = project_images_cylindrical(images, config)
    summary["focal_lengths"] = {
        image.name: round(projected.focal_length, 6)
        for image, projected in zip(images, projected_images)
    }

    try:
        pairwise_transforms, step_summaries, pairwise_translations = _estimate_pairwise_cylindrical_transforms(
            projected_images,
            config,
            output_path,
        )
        summary["steps"].extend(step_summaries)
        summary["pairwise_translations"] = pairwise_translations
        transforms, valid_indices = _accumulate_cylindrical_transforms(projected_images, pairwise_transforms, reference_idx)
        summary["valid_image_indices"] = valid_indices
        summary["global_transforms"] = {
            projected_images[idx].name: transforms[idx].round(6).tolist() for idx in valid_indices
        }
        panorama_full, panorama_cropped, composition_debug = _compose_cylindrical_panorama(
            projected_images,
            transforms,
            reference_idx,
            config,
        )
    except ValueError as exc:
        summary["status"] = "failed"
        summary["failure_reason"] = str(exc)
        summary["failed_stage"] = "global_composition"
        return summary

    write_image(output_path / "panorama_final.jpg", panorama_cropped)
    summary["composition"] = composition_debug
    summary["canvas_scale"] = composition_debug["canvas_scale"]
    summary["valid_image_indices"] = composition_debug["valid_image_indices"]
    summary["blend_order"] = composition_debug["blend_order"]
    summary["panorama_full_shape"] = list(panorama_full.shape)
    summary["final_panorama_shape"] = list(panorama_cropped.shape)
    summary["status"] = "ok"
    return summary


def run_panorama(images: List[ImageData], config: dict, output_dir: str | Path) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    projection_mode = str(config.get("projection", {}).get("mode", "planar")).lower()
    if projection_mode == "cylindrical":
        return _run_cylindrical_panorama(images, config, output_path)
    return _run_planar_panorama(images, config, output_path)
