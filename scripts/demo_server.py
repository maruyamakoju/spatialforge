"""RailScan Demo Server — Local inference with YOLOv8m.

Serves the demo site and provides real-time YOLO inference endpoints.
Designed for RTX 4090 local demo, with minimal GPU load.

Usage:
    python scripts/demo_server.py [--port 8765] [--model models/railscan-yolov8m.pt]

Endpoints:
    GET  /health               → {"status":"ok","model":"railscan-yolov8m","gpu":"..."}
    POST /v1/depth/visualize   → overlay JPEG with detection headers
    POST /v1/detect            → JSON detection results
    GET  /*                    → static files from site/
"""

import argparse
import io
import json
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lazy-load YOLO to avoid slow startup if just serving static
_model = None
_model_path = None

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
    "major":    (11, 158, 245),   # #f59e0b in BGR
    "minor":    (246, 130, 59),   # #3b82f6 in BGR
    "info":     (128, 114, 107),  # #6b7280 in BGR
}


def get_model():
    """Lazy-load YOLO model on first inference request."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        print(f"[model] Loading {_model_path}...")
        _model = YOLO(str(_model_path))
        # Warmup with a dummy image (low GPU load)
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _model(dummy, conf=0.5, imgsz=64, verbose=False)
        print("[model] Ready for inference")
    return _model


def run_inference(img_bytes):
    """Run YOLOv8m inference on image bytes. Returns (detections, speed_ms, img)."""
    # Decode image
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0, None

    model = get_model()
    t0 = time.perf_counter()
    results = model(img, conf=0.25, imgsz=640, verbose=False)
    speed_ms = (time.perf_counter() - t0) * 1000

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
                "cls_ja": CLASS_LABELS_JA.get(cls_name, cls_name),
                "conf": round(conf, 3),
                "sev": sev,
                "area": int((x2 - x1) * (y2 - y1)),
            })

    return detections, round(speed_ms, 1), img


def draw_overlay(img, detections):
    """Draw bounding boxes with labels on image. Returns JPEG bytes."""
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

        # Label background
        label = f"{ja_label} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(overlay, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(overlay, label, (x1 + 4, label_y - 2), font, font_scale, (255, 255, 255), font_thickness)

        # Corner markers
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        if corner_len > 2:
            ct = thickness + 1
            cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), color, ct)
            cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), color, ct)
            cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), color, ct)
            cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), color, ct)
            cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), color, ct)
            cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), color, ct)
            cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), color, ct)
            cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), color, ct)

    _, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpeg.tobytes()


def generate_depth_colormap(img):
    """Generate a fake depth colormap from image (for visual demo)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 15, 80, 80)
    smooth = 255 - smooth
    h, w = smooth.shape
    gradient = np.linspace(60, 200, h).reshape(-1, 1).astype(np.uint8)
    gradient = np.broadcast_to(gradient, (h, w))
    blended = cv2.addWeighted(smooth, 0.6, gradient, 0.4, 0)
    colored = cv2.applyColorMap(blended, cv2.COLORMAP_TURBO)
    return colored


def parse_multipart(content_type, body):
    """Parse multipart/form-data body. Returns dict of {field_name: bytes}."""
    # Extract boundary from content type
    boundary = None
    for part in content_type.split(';'):
        part = part.strip()
        if part.startswith('boundary='):
            boundary = part[len('boundary='):]
            break
    if not boundary:
        return {}

    boundary_bytes = boundary.encode('utf-8')
    parts = body.split(b'--' + boundary_bytes)
    fields = {}

    for part in parts:
        if part in (b'', b'--\r\n', b'--'):
            continue
        part = part.strip(b'\r\n')
        if b'\r\n\r\n' not in part:
            continue
        headers_raw, data = part.split(b'\r\n\r\n', 1)
        # Remove trailing boundary marker
        if data.endswith(b'\r\n'):
            data = data[:-2]
        if data.endswith(b'--'):
            data = data[:-2]

        # Parse Content-Disposition
        headers_str = headers_raw.decode('utf-8', errors='replace')
        name = None
        for header_line in headers_str.split('\r\n'):
            if 'Content-Disposition' in header_line:
                for token in header_line.split(';'):
                    token = token.strip()
                    if token.startswith('name='):
                        name = token[5:].strip('"')
        if name:
            fields[name] = data

    return fields


class DemoHandler(SimpleHTTPRequestHandler):
    """HTTP handler: static files + inference endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "site"), **kwargs)

    def log_message(self, format, *args):
        # Suppress noisy static file logs, keep API logs
        path = args[0].split()[1] if args else ''
        if path.startswith('/v1/') or path == '/health':
            super().log_message(format, *args)

    def end_headers(self):
        # CORS headers for all responses
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Expose-Headers',
                         'X-Depth-Near-M, X-Depth-Far-M, X-Processing-Ms, '
                         'X-Model, X-Anomaly-Count, X-Confidence')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/health':
            self._handle_health()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/v1/depth/visualize':
            self._handle_visualize()
        elif path == '/v1/detect':
            self._handle_detect()
        else:
            self.send_error(404, 'Not Found')

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length > 20 * 1024 * 1024:  # 20MB limit
            return None
        return self.rfile.read(length)

    def _extract_image(self):
        """Extract image bytes from multipart or raw body."""
        content_type = self.headers.get('Content-Type', '')
        body = self._read_body()
        if body is None:
            return None

        if 'multipart/form-data' in content_type:
            fields = parse_multipart(content_type, body)
            return fields.get('image') or fields.get('file')
        elif content_type.startswith('image/'):
            return body
        else:
            return body  # Try raw

    def _handle_health(self):
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        data = {
            "status": "ok",
            "model": Path(_model_path).stem,
            "gpu": gpu_name,
            "classes": len(CLASS_NAMES),
        }
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _handle_visualize(self):
        """POST /v1/depth/visualize — returns overlay JPEG with detection headers."""
        img_bytes = self._extract_image()
        if not img_bytes:
            self.send_error(400, 'No image provided')
            return

        detections, speed_ms, img = run_inference(img_bytes)
        if img is None:
            self.send_error(400, 'Could not decode image')
            return

        # Draw overlay
        overlay_bytes = draw_overlay(img, detections)

        # Compute summary stats
        max_conf = max((d["conf"] for d in detections), default=0)
        max_sev = "info"
        for d in detections:
            if d["sev"] == "critical":
                max_sev = "critical"
                break
            elif d["sev"] == "major":
                max_sev = "major"

        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(overlay_bytes))
        self.send_header('X-Depth-Near-M', '2.0')
        self.send_header('X-Depth-Far-M', '50.0')
        self.send_header('X-Processing-Ms', str(speed_ms))
        self.send_header('X-Model', 'YOLOv8m-railscan')
        self.send_header('X-Anomaly-Count', str(len(detections)))
        self.send_header('X-Confidence', str(max_conf))
        self.send_header('X-Max-Severity', max_sev)
        self.end_headers()
        self.wfile.write(overlay_bytes)
        print(f"  → {len(detections)} detections, {speed_ms}ms")

    def _handle_detect(self):
        """POST /v1/detect — returns JSON detection results."""
        img_bytes = self._extract_image()
        if not img_bytes:
            self.send_error(400, 'No image provided')
            return

        detections, speed_ms, img = run_inference(img_bytes)
        if img is None:
            self.send_error(400, 'Could not decode image')
            return

        h, w = img.shape[:2]
        result = {
            "detections": detections,
            "count": len(detections),
            "speed_ms": speed_ms,
            "image_size": [w, h],
            "model": Path(_model_path).stem,
        }
        body = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)
        print(f"  → {len(detections)} detections, {speed_ms}ms (JSON)")


def main():
    global _model_path
    parser = argparse.ArgumentParser(description='RailScan Demo Server')
    parser.add_argument('--port', type=int, default=8765, help='Server port (default: 8765)')
    parser.add_argument('--model', type=str,
                        default=str(PROJECT_ROOT / 'models' / 'railscan-yolov8m.pt'),
                        help='Path to YOLO model')
    parser.add_argument('--preload', action='store_true',
                        help='Preload model at startup (uses GPU immediately)')
    args = parser.parse_args()

    _model_path = args.model
    if not Path(_model_path).exists():
        print(f"ERROR: Model not found at {_model_path}")
        sys.exit(1)

    if args.preload:
        get_model()

    server = HTTPServer(('0.0.0.0', args.port), DemoHandler)
    print(f"\n{'='*60}")
    print(f"  RailScan Demo Server")
    print(f"  http://localhost:{args.port}/rail-demo.html")
    print(f"{'='*60}")
    print(f"  Model:  {Path(_model_path).name}")
    print(f"  Port:   {args.port}")
    print(f"  Static: {PROJECT_ROOT / 'site'}")
    print(f"{'='*60}")
    print(f"  Endpoints:")
    print(f"    GET  /health             → API health check")
    print(f"    POST /v1/depth/visualize → overlay JPEG + headers")
    print(f"    POST /v1/detect          → JSON detection results")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
