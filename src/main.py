from __future__ import annotations

import argparse
from pathlib import Path

from .preprocess import load_images
from .stitcher import run_panorama
from .utils import ensure_dir, load_config, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Panorama stitching pipeline using ORB features.")
    parser.add_argument("--input_dir", default="input", help="Directory containing input images.")
    parser.add_argument("--output_dir", default="output", help="Directory used to save outputs.")
    parser.add_argument("--feature", default=None, help="Override feature extractor from config.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.feature:
        config["feature"]["name"] = args.feature

    output_dir = ensure_dir(args.output_dir)
    images = load_images(args.input_dir, config)
    summary = run_panorama(images, config, output_dir)
    save_json(Path(output_dir) / "run_summary.json", summary)
    if summary.get("status") != "ok":
        raise SystemExit(summary.get("failure_reason", "Panorama stitching failed."))


if __name__ == "__main__":
    main()
