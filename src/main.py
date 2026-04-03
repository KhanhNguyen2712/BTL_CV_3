from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .preprocess import discover_input_sequences, load_images
from .stitcher import run_panorama
from .utils import ensure_dir, load_config, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Panorama stitching pipeline using ORB or SIFT features.")
    parser.add_argument("--input_dir", default="input", help="Directory containing input images.")
    parser.add_argument("--output_dir", default="output", help="Directory used to save outputs.")
    parser.add_argument("--feature", default=None, help="Override feature extractor from config.")
    parser.add_argument("--projection", default=None, help="Override projection mode from config.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser


def _resolve_dataset_output_dir(base_output_dir: Path, dataset_name: str, total_datasets: int) -> Path:
    if total_datasets == 1 and base_output_dir.name == dataset_name:
        return ensure_dir(base_output_dir)
    return ensure_dir(base_output_dir / dataset_name)


def _deep_update(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _config_for_sequence(config: dict, sequence_name: str) -> dict:
    dataset_overrides = config.get("dataset_overrides", {})
    sequence_overrides = dataset_overrides.get(sequence_name, {})
    if not sequence_overrides:
        return copy.deepcopy(config)
    merged_config = _deep_update(config, sequence_overrides)
    merged_config.pop("dataset_overrides", None)
    return merged_config


def _apply_runtime_defaults(config: dict) -> dict:
    feature_name = str(config.get("feature", {}).get("name", "orb")).lower()
    projection_mode = str(config.get("projection", {}).get("mode", "planar")).lower()
    if projection_mode == "cylindrical" and feature_name != "sift":
        raise ValueError("Projection mode 'cylindrical' is only supported with feature extractor 'sift'.")

    if feature_name != "sift":
        return config

    sift_defaults: dict = {
        "matching": {
            "matcher": "flann",
            "norm": "l2",
            "ratio": 0.75,
            "max_good_matches": 200,
            "flann_trees": 5,
            "flann_checks": 50,
        }
    }
    if projection_mode == "cylindrical":
        sift_defaults = _deep_update(
            sift_defaults,
            {
                "matching": {
                    "symmetry_check": False,
                    "fallback_cross_check": False,
                    "pairwise_min_occupied_cells": 0,
                }
            },
        )
    return _deep_update(config, sift_defaults)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.feature:
        config["feature"]["name"] = args.feature
    if args.projection:
        config["projection"]["mode"] = args.projection
    config = _apply_runtime_defaults(config)

    base_output_dir = ensure_dir(args.output_dir)
    input_sequences = discover_input_sequences(args.input_dir)
    batch_summary = {
        "input_root": str(Path(args.input_dir)),
        "output_root": str(base_output_dir),
        "datasets": [],
    }
    failures: list[str] = []

    for sequence in input_sequences:
        dataset_output_dir = _resolve_dataset_output_dir(base_output_dir, sequence.name, len(input_sequences))
        dataset_config = _apply_runtime_defaults(_config_for_sequence(config, sequence.name))
        images = load_images(sequence.path, dataset_config)
        summary = run_panorama(images, dataset_config, dataset_output_dir)
        summary["input"] = {"name": sequence.name, "path": sequence.path}
        summary["applied_capture_direction"] = dataset_config.get("homography", {}).get("capture_direction", "")
        save_json(dataset_output_dir / "run_summary.json", summary)
        batch_summary["datasets"].append(
            {
                "name": sequence.name,
                "input_dir": sequence.path,
                "output_dir": str(dataset_output_dir),
                "status": summary.get("status"),
                "failure_reason": summary.get("failure_reason"),
                "capture_direction": summary["applied_capture_direction"],
            }
        )
        if summary.get("status") != "ok":
            failures.append(f"{sequence.name}: {summary.get('failure_reason', 'Panorama stitching failed.')}")

    if len(input_sequences) > 1:
        save_json(base_output_dir / "batch_summary.json", batch_summary)

    if failures:
        raise SystemExit("Panorama stitching failed for dataset(s): " + "; ".join(failures))


if __name__ == "__main__":
    main()
