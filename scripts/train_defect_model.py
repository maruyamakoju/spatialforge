#!/usr/bin/env python3
"""Train YOLOv8 defect detection model for RailScan.

Fine-tunes a YOLOv8 model on the prepared rail defect dataset.
Produces a .pt checkpoint ready for deployment.

Usage:
    # Prepare dataset first
    python scripts/prepare_dataset.py --output ./datasets/railscan

    # Train model
    python scripts/train_defect_model.py --data ./datasets/railscan/data.yaml

    # Train with specific settings
    python scripts/train_defect_model.py \\
        --data ./datasets/railscan/data.yaml \\
        --model yolov8m.pt \\
        --epochs 100 \\
        --batch 16 \\
        --imgsz 640 \\
        --device 0

    # Export to ONNX for edge deployment
    python scripts/train_defect_model.py --export --weights runs/detect/railscan/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default training hyperparameters optimized for rail defect detection.
# These are tuned for small defects on high-resolution track images.
DEFAULT_HYPERPARAMS = {
    "lr0": 0.01,           # Initial learning rate
    "lrf": 0.01,           # Final learning rate (lr0 * lrf)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,            # Box loss gain
    "cls": 0.5,            # Classification loss gain
    "dfl": 1.5,            # Distribution focal loss gain
    "hsv_h": 0.015,        # HSV hue augmentation
    "hsv_s": 0.7,          # HSV saturation augmentation
    "hsv_v": 0.4,          # HSV value augmentation
    "degrees": 5.0,        # Rotation augmentation (small for tracks)
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0,
    "flipud": 0.5,         # Vertical flip (tracks can be upside down in camera)
    "fliplr": 0.5,         # Horizontal flip
    "mosaic": 1.0,         # Mosaic augmentation
    "mixup": 0.1,          # Mixup augmentation
    "copy_paste": 0.1,     # Copy-paste augmentation
}


def train(
    data_yaml: Path,
    model_name: str = "yolov8m.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "railscan",
    resume: bool = False,
) -> Path:
    """Train YOLOv8 model on rail defect dataset.

    Args:
        data_yaml: Path to data.yaml file.
        model_name: Base model to fine-tune from.
        epochs: Number of training epochs.
        batch_size: Batch size.
        imgsz: Input image size.
        device: CUDA device (0, 1, cpu).
        project: Project directory for saving results.
        name: Experiment name.
        resume: Resume from last checkpoint.

    Returns:
        Path to best model weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics is required. Install with:\n"
            "  pip install ultralytics"
        )
        raise

    logger.info("=" * 60)
    logger.info("RailScan Defect Detection — Model Training")
    logger.info("=" * 60)
    logger.info("  Base model: %s", model_name)
    logger.info("  Dataset:    %s", data_yaml)
    logger.info("  Epochs:     %d", epochs)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Image size: %d", imgsz)
    logger.info("  Device:     %s", device)
    logger.info("=" * 60)

    # Load base model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        patience=20,          # Early stopping patience
        save_period=10,       # Save checkpoint every N epochs
        plots=True,           # Generate training plots
        verbose=True,
        **DEFAULT_HYPERPARAMS,
    )

    # Path to best weights
    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        logger.info("Best model saved to: %s", best_weights)

        # Copy to models/ directory for easy deployment
        deploy_path = Path("models") / "railscan-yolov8m.pt"
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, deploy_path)
        logger.info("Deployed model copied to: %s", deploy_path)
    else:
        logger.warning("Best weights not found at expected path: %s", best_weights)

    return best_weights


def evaluate(
    weights_path: Path,
    data_yaml: Path,
    device: str = "0",
) -> None:
    """Evaluate trained model on test set."""
    from ultralytics import YOLO

    logger.info("Evaluating model: %s", weights_path)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        device=device,
        split="test",
        plots=True,
        verbose=True,
    )

    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info("  mAP50:    %.4f", metrics.box.map50)
    logger.info("  mAP50-95: %.4f", metrics.box.map)
    logger.info("  Precision: %.4f", metrics.box.mp)
    logger.info("  Recall:    %.4f", metrics.box.mr)
    logger.info("=" * 60)


def export_model(
    weights_path: Path,
    format: str = "onnx",
    imgsz: int = 640,
) -> Path:
    """Export trained model to deployment format.

    Supported formats: onnx, torchscript, tflite, coreml, engine (TensorRT)
    """
    from ultralytics import YOLO

    logger.info("Exporting %s to %s format", weights_path, format)

    model = YOLO(str(weights_path))
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=format in ("engine",),  # FP16 for TensorRT
        simplify=True,
    )

    logger.info("Exported model: %s", export_path)
    return Path(export_path)


def main():
    parser = argparse.ArgumentParser(description="Train RailScan defect detection model")
    parser.add_argument(
        "--data", type=Path, default=Path("./datasets/railscan/data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--resume", action="store_true")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--weights", type=Path, help="Model weights for eval/export")

    # Export
    parser.add_argument("--export", action="store_true", help="Export model")
    parser.add_argument("--export-format", type=str, default="onnx")

    args = parser.parse_args()

    if args.export:
        if not args.weights:
            args.weights = Path("runs/detect/railscan/weights/best.pt")
        export_model(args.weights, format=args.export_format, imgsz=args.imgsz)
    elif args.eval:
        if not args.weights:
            args.weights = Path("runs/detect/railscan/weights/best.pt")
        evaluate(args.weights, args.data, device=args.device)
    else:
        train(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
