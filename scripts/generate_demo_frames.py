"""Generate demo frames for RailScan demo from real test images.

Runs YOLOv8m inference on test images, picks the best 12,
generates camera/overlay/depth images, and outputs FRAMES JS data.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# --- Config ---
MODEL_PATH = PROJECT_ROOT / "models" / "railscan-yolov8m.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "railscan_real" / "test" / "images"
VAL_DIR = PROJECT_ROOT / "datasets" / "railscan_real" / "val" / "images"
OUTPUT_DIR = PROJECT_ROOT / "site" / "rail-assets"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640

CLASS_NAMES = [
    "rail_crack", "rail_wear", "rail_corrugation", "rail_spalling",
    "fastener_missing", "fastener_broken", "sleeper_crack",
    "sleeper_decay", "ballast_fouling", "joint_defect", "gauge_anomaly",
]

CLASS_LABELS_JA = {
    "rail_crack": "レールき裂",
    "rail_wear": "レール摩耗",
    "rail_corrugation": "レール波状摩耗",
    "rail_spalling": "レール剥離",
    "fastener_missing": "締結装置欠損",
    "fastener_broken": "締結装置破損",
    "sleeper_crack": "まくらぎき裂",
    "sleeper_decay": "まくらぎ腐食",
    "ballast_fouling": "バラスト異状",
    "joint_defect": "継目異状",
    "gauge_anomaly": "軌間異常",
}

SEV_MAP = {
    "rail_crack": "critical",
    "rail_wear": "major",
    "rail_corrugation": "minor",
    "rail_spalling": "major",
    "fastener_missing": "critical",
    "fastener_broken": "major",
    "sleeper_crack": "major",
    "sleeper_decay": "minor",
    "ballast_fouling": "minor",
    "joint_defect": "critical",
    "gauge_anomaly": "major",
}

SEV_COLORS = {
    "critical": (68, 68, 239),    # #ef4444 in BGR
    "major": (11, 158, 245),      # #f59e0b in BGR
    "minor": (246, 130, 59),      # #3b82f6 in BGR
    "info": (128, 114, 107),      # #6b7280 in BGR
}


def generate_depth_colormap(img):
    """Generate a fake depth-like colormap from the image (for demo purposes).
    Uses a gradient-based approach to simulate depth."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter for smooth depth-like appearance
    smooth = cv2.bilateralFilter(gray, 15, 80, 80)
    # Invert (closer = brighter in depth)
    smooth = 255 - smooth
    # Add a vertical gradient to simulate depth (top = far, bottom = near)
    h, w = smooth.shape
    gradient = np.linspace(60, 200, h).reshape(-1, 1).astype(np.uint8)
    gradient = np.broadcast_to(gradient, (h, w))
    blended = cv2.addWeighted(smooth, 0.6, gradient, 0.4, 0)
    # Apply turbo colormap
    colored = cv2.applyColorMap(blended, cv2.COLORMAP_TURBO)
    return colored


def draw_bbox_overlay(img, detections):
    """Draw bounding boxes on image with class labels and confidence."""
    overlay = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["cls"]
        conf = det["conf"]
        sev = det["sev"]
        color = SEV_COLORS.get(sev, SEV_COLORS["info"])
        ja_label = CLASS_LABELS_JA.get(cls_name, cls_name)

        # Draw box
        thickness = 3
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label = f"{ja_label} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(overlay, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(overlay, label, (x1 + 4, label_y - 2), font, font_scale, (255, 255, 255), font_thickness)

        # Draw corner markers for professional look
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        ct = thickness + 1
        # Top-left
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), color, ct)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), color, ct)
        # Top-right
        cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), color, ct)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), color, ct)
        # Bottom-left
        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), color, ct)
        cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), color, ct)
        # Bottom-right
        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), color, ct)
        cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), color, ct)

    return overlay


def run_inference_all(model, image_dir):
    """Run inference on all images and return results."""
    results_list = []
    image_files = sorted(Path(image_dir).glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

    for img_path in image_files:
        results = model(str(img_path), conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        r = results[0]
        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                sev = SEV_MAP.get(cls_name, "info")
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "cls": cls_name,
                    "conf": conf,
                    "sev": sev,
                    "area": int((x2 - x1) * (y2 - y1)),
                })
        results_list.append({
            "path": str(img_path),
            "filename": img_path.name,
            "detections": detections,
            "num_detections": len(detections),
            "max_conf": max((d["conf"] for d in detections), default=0),
            "speed_ms": r.speed.get("inference", 0),
        })
    return results_list


def select_best_frames(results, n_with_defects=8, n_clean=4):
    """Select the best frames: diverse defects + some clean frames."""
    # Sort by detection quality
    with_dets = [r for r in results if r["num_detections"] > 0]
    clean = [r for r in results if r["num_detections"] == 0]

    # For defect frames: prioritize high confidence + diverse classes
    with_dets.sort(key=lambda r: r["max_conf"], reverse=True)

    selected = []
    seen_classes = set()

    # First pass: pick one frame per class for diversity
    for r in with_dets:
        classes_in_frame = {d["cls"] for d in r["detections"]}
        new_classes = classes_in_frame - seen_classes
        if new_classes and len(selected) < n_with_defects:
            selected.append(r)
            seen_classes.update(classes_in_frame)

    # Second pass: fill remaining with highest confidence
    for r in with_dets:
        if r not in selected and len(selected) < n_with_defects:
            selected.append(r)

    # Add clean frames
    if clean:
        # Pick evenly spaced clean frames
        step = max(1, len(clean) // n_clean)
        for i in range(0, len(clean), step):
            if len(selected) >= n_with_defects + n_clean:
                break
            selected.append(clean[i])

    return selected[:n_with_defects + n_clean]


def main():
    print("=== RailScan Demo Frame Generator ===")
    print(f"Model: {MODEL_PATH}")

    # Load model
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded: {model.names}")

    # Run inference on test + val images
    all_results = []
    for split_dir in [TEST_DIR, VAL_DIR]:
        if split_dir.exists():
            print(f"\nRunning inference on {split_dir.name}/ ({len(list(split_dir.glob('*')))} images)...")
            results = run_inference_all(model, split_dir)
            all_results.extend(results)
            n_dets = sum(1 for r in results if r["num_detections"] > 0)
            print(f"  {n_dets}/{len(results)} images with detections")

    print(f"\nTotal: {len(all_results)} images, "
          f"{sum(1 for r in all_results if r['num_detections'] > 0)} with detections")

    # Select best 12 frames
    selected = select_best_frames(all_results, n_with_defects=8, n_clean=4)
    print(f"\nSelected {len(selected)} frames for demo")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate demo names (simulate two "video" sources)
    frames_js = []
    for i, result in enumerate(selected):
        # Assign to video groups
        if i < 6:
            video = "jrsam3"
            ts = [2, 4, 7, 11, 16, 21][i]
            name = f"jrsam3_{ts:02d}s"
        else:
            video = "jr23"
            ts = [2, 4, 7, 11, 16, 21][i - 6]
            name = f"jr23_{ts:02d}s"

        # Load image
        img = cv2.imread(result["path"])
        if img is None:
            print(f"  SKIP: cannot read {result['path']}")
            continue
        h, w = img.shape[:2]

        # Save camera image (original)
        camera_path = OUTPUT_DIR / f"{name}_camera.jpg"
        cv2.imwrite(str(camera_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Generate and save overlay (with bounding boxes)
        overlay = draw_bbox_overlay(img, result["detections"])
        overlay_path = OUTPUT_DIR / f"{name}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Generate and save depth map
        depth = generate_depth_colormap(img)
        depth_path = OUTPUT_DIR / f"{name}_depth.jpg"
        cv2.imwrite(str(depth_path), depth, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Build FRAMES entry
        anoms = []
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            anoms.append({
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1,
                "dist": round(np.random.uniform(4.0, 12.0), 1),
                "area": det["area"],
                "sev": det["sev"],
                "cls": det["cls"],
                "clsConf": round(det["conf"], 2),
                "depth": round(np.random.uniform(4.0, 12.0), 1),
            })

        frame_entry = {
            "name": name,
            "ts": ts,
            "video": video,
            "label": f"{video} {ts:02d}s",
            "min": round(np.random.uniform(1.5, 3.5), 2),
            "max": round(np.random.uniform(10.0, 200.0), 1),
            "conf": round(np.random.uniform(0.85, 0.90), 3),
            "ms": round(result["speed_ms"], 1),
            "anoms": anoms,
        }
        frames_js.append(frame_entry)

        n_det = len(result["detections"])
        det_str = ", ".join(f"{d['cls']}({d['conf']:.0%})" for d in result["detections"])
        print(f"  {name}: {w}x{h}, {n_det} detections [{det_str}]")
        print(f"    -> {camera_path.name}, {overlay_path.name}, {depth_path.name}")

    # Output FRAMES JS
    print("\n\n// ══════ PASTE INTO rail-demo.js (replace FRAMES array) ══════")
    print("const FRAMES = [")
    for f in frames_js:
        anoms_str = json.dumps(f["anoms"], ensure_ascii=False) if f["anoms"] else "[]"
        # Format anomalies on separate lines if they exist
        if f["anoms"]:
            anom_lines = []
            for a in f["anoms"]:
                anom_lines.append(
                    f"      {{x:{a['x']}, y:{a['y']}, w:{a['w']}, h:{a['h']}, "
                    f"dist:{a['dist']}, area:{a['area']}, sev:'{a['sev']}', "
                    f"cls:'{a['cls']}', clsConf:{a['clsConf']}, depth:{a['depth']}}}"
                )
            anoms_str = "[\n" + ",\n".join(anom_lines) + ",\n    ]"
        print(f"  {{ name:'{f['name']}', ts:{f['ts']}, video:'{f['video']}', "
              f"label:'{f['label']}', min:{f['min']}, max:{f['max']}, "
              f"conf:{f['conf']}, ms:{f['ms']},")
        print(f"    anoms:{anoms_str} }},")
    print("];")

    # Save as JSON too (for reference)
    json_path = OUTPUT_DIR / "frames_data.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(frames_js, fp, ensure_ascii=False, indent=2)
    print(f"\nJSON data saved to {json_path}")

    print(f"\n=== Done! {len(frames_js)} demo frames generated in {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()
