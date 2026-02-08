#!/usr/bin/env python3
"""Download and cache depth estimation models for SpatialForge.

Usage:
    python scripts/setup_models.py                  # Download default (giant)
    python scripts/setup_models.py --model all      # Download all sizes
    python scripts/setup_models.py --model small    # Download specific size
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small",
    "base": "depth-anything/Depth-Anything-V2-Base",
    "large": "depth-anything/Depth-Anything-V2-Large",
    "giant": "depth-anything/Depth-Anything-V2-Giant",
}

# DA3 models (try these first when available)
DA3_MODELS = {
    "small": "DepthAnything/Depth-Anything-3-Small",
    "base": "DepthAnything/Depth-Anything-3-Base",
    "large": "DepthAnything/Depth-Anything-3-Large",
    "giant": "DepthAnything/Depth-Anything-3-Giant",
}


def download_model(name: str, cache_dir: Path) -> bool:
    """Download a depth model via HuggingFace transformers."""
    try:
        from transformers import pipeline

        # Try DA3 first
        da3_repo = DA3_MODELS.get(name)
        if da3_repo:
            try:
                logger.info("Trying DA3-%s: %s", name, da3_repo)
                pipe = pipeline(
                    "depth-estimation",
                    model=da3_repo,
                    device="cpu",
                )
                logger.info("Successfully downloaded DA3-%s", name)
                del pipe
                return True
            except Exception as e:
                logger.info("DA3-%s not available (%s), falling back to DAv2", name, e)

        # Fallback to DAv2
        repo = MODELS.get(name)
        if not repo:
            logger.error("Unknown model: %s", name)
            return False

        logger.info("Downloading DAv2-%s: %s", name, repo)
        pipe = pipeline(
            "depth-estimation",
            model=repo,
            device="cpu",
        )
        logger.info("Successfully downloaded DAv2-%s", name)
        del pipe
        return True

    except Exception as e:
        logger.error("Failed to download %s: %s", name, e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download depth estimation models")
    parser.add_argument(
        "--model",
        choices=["small", "base", "large", "giant", "all"],
        default="giant",
        help="Which model size to download (default: giant)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./models"),
        help="Cache directory for models",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    success = 0
    for name in models:
        if download_model(name, args.cache_dir):
            success += 1

    logger.info("Downloaded %d/%d models", success, len(models))
    if success < len(models):
        sys.exit(1)


if __name__ == "__main__":
    main()
