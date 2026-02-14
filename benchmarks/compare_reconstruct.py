#!/usr/bin/env python3
"""Run reconstruct benchmarks across backends and generate a comparison report."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import git_commit_short, system_info, utc_timestamp, write_json

_ALLOWED_BACKENDS = {"legacy", "tsdf", "da3"}


def _parse_backends(raw: str) -> list[str]:
    ordered: list[str] = []
    for token in raw.split(","):
        backend = token.strip().lower()
        if not backend:
            continue
        if backend not in _ALLOWED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Allowed: {sorted(_ALLOWED_BACKENDS)}")
        if backend not in ordered:
            ordered.append(backend)
    if not ordered:
        raise ValueError("Provide at least one backend")
    return ordered


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_command(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _extract_quality_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    records = [item for item in payload.get("results", []) if isinstance(item, dict)]
    ok = [item for item in records if "error" not in item]

    coverage: list[float] = []
    abs_depth: list[float] = []
    inlier_ratio: list[float] = []

    for item in ok:
        fit = item.get("rendered_depth_fit", {})
        cov_mean = ((fit.get("coverage") or {}).get("mean"))
        abs_mean = ((fit.get("abs_depth_error_m") or {}).get("mean"))
        inlier_mean = ((fit.get("inlier_ratio") or {}).get("mean"))
        if cov_mean is not None:
            coverage.append(float(cov_mean))
        if abs_mean is not None:
            abs_depth.append(float(abs_mean))
        if inlier_mean is not None:
            inlier_ratio.append(float(inlier_mean))

    return {
        "num_sequences": len(records),
        "num_success": len(ok),
        "num_failed": len(records) - len(ok),
        "coverage_mean": _mean_or_none(coverage),
        "abs_depth_error_mean_m": _mean_or_none(abs_depth),
        "inlier_ratio_mean": _mean_or_none(inlier_ratio),
    }


def _extract_bench_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    records = [item for item in payload.get("results", []) if isinstance(item, dict)]
    ok = [item for item in records if "error" not in item]

    lat_p50: list[float] = []
    lat_p95: list[float] = []
    throughput_fps: list[float] = []
    peak_vram: list[float] = []

    for item in ok:
        latency = item.get("latency_ms") or {}
        p50 = latency.get("p50")
        p95 = latency.get("p95")
        mean_ms = latency.get("mean")
        frame_count = item.get("frame_count")
        peak = item.get("peak_vram_gb")
        if p50 is not None:
            lat_p50.append(float(p50))
        if p95 is not None:
            lat_p95.append(float(p95))
        if mean_ms is not None and frame_count:
            throughput_fps.append(float(frame_count) / (float(mean_ms) / 1000.0))
        if peak is not None:
            peak_vram.append(float(peak))

    return {
        "num_cases": len(records),
        "num_success": len(ok),
        "num_failed": len(records) - len(ok),
        "latency_p50_ms_mean": _mean_or_none(lat_p50),
        "latency_p95_ms_mean": _mean_or_none(lat_p95),
        "throughput_fps_mean": _mean_or_none(throughput_fps),
        "peak_vram_gb_max": max(peak_vram) if peak_vram else None,
    }


def _has_backend_failures(quality_payload: dict[str, Any], bench_payload: dict[str, Any]) -> bool:
    quality_failed = int((quality_payload.get("summary") or {}).get("num_failed", 0))
    bench_errors = sum(1 for item in bench_payload.get("results", []) if isinstance(item, dict) and "error" in item)
    return quality_failed > 0 or bench_errors > 0


def _delta_or_none(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_signed(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def _build_markdown(
    *,
    report_payload: dict[str, Any],
    backend_rows: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, float | None]],
) -> str:
    config = report_payload["config"]
    system = report_payload["system"]
    baseline = config["backends"][0]

    lines: list[str] = []
    lines.append("# Reconstruct Backend Comparison")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report_payload['timestamp_utc']}`")
    lines.append(f"- Git commit: `{report_payload['git_commit']}`")
    lines.append(f"- Device: `{config['device']}`")
    lines.append(f"- DType: `{config['dtype']}`")
    lines.append(f"- Depth backend: `{config['depth_backend']}`")
    lines.append(f"- Reconstruct backends: `{', '.join(config['backends'])}`")
    lines.append(f"- GPU: `{system.get('gpu_name')}`")
    lines.append("")
    lines.append("## Backend Summary")
    lines.append("")
    lines.append(
        "| Backend | Coverage mean | Abs depth error mean (m) | Inlier ratio mean | "
        "Latency p50 (ms) | Latency p95 (ms) | Throughput mean (fps) | Peak VRAM max (GB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for backend, row in backend_rows.items():
        quality = row["quality_summary"]
        bench = row["bench_summary"]
        lines.append(
            "| "
            f"{backend} | "
            f"{_fmt_float(quality['coverage_mean'])} | "
            f"{_fmt_float(quality['abs_depth_error_mean_m'])} | "
            f"{_fmt_float(quality['inlier_ratio_mean'])} | "
            f"{_fmt_float(bench['latency_p50_ms_mean'], 3)} | "
            f"{_fmt_float(bench['latency_p95_ms_mean'], 3)} | "
            f"{_fmt_float(bench['throughput_fps_mean'], 3)} | "
            f"{_fmt_float(bench['peak_vram_gb_max'], 3)} |"
        )

    lines.append("")
    lines.append(f"## Deltas vs `{baseline}`")
    lines.append("")
    if len(config["backends"]) <= 1:
        lines.append("- Single-backend run; no delta rows.")
    else:
        for backend in config["backends"][1:]:
            delta = deltas[backend]
            lines.append(
                f"- `{backend}`: "
                f"coverage `{_fmt_signed(delta['coverage_mean'])}`, "
                f"abs_depth_error_mean_m `{_fmt_signed(delta['abs_depth_error_mean_m'])}`, "
                f"inlier_ratio_mean `{_fmt_signed(delta['inlier_ratio_mean'])}`, "
                f"latency_p50_ms `{_fmt_signed(delta['latency_p50_ms_mean'], 3)}`, "
                f"latency_p95_ms `{_fmt_signed(delta['latency_p95_ms_mean'], 3)}`, "
                f"throughput_fps `{_fmt_signed(delta['throughput_fps_mean'], 3)}`."
            )

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for backend, row in backend_rows.items():
        lines.append(
            f"- `{backend}`: quality=`{row['quality_file']}`, bench=`{row['bench_file']}`"
        )
    lines.append("- consolidated JSON: `reconstruct_compare.json`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reconstruct compare report across backends")
    parser.add_argument("--dataset-dir", default=None, help="Optional sequence root for quality runs")
    parser.add_argument("--depth-backend", choices=["hf", "da3"], default="da3")
    parser.add_argument("--backends", default="legacy,tsdf")
    parser.add_argument("--quality", choices=["draft", "standard", "high"], default="standard")
    parser.add_argument("--output-format", choices=["pointcloud", "gaussian", "mesh"], default="pointcloud")
    parser.add_argument("--frame-counts", default="30,60,120")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--render-downscale", type=float, default=0.5)
    parser.add_argument("--depth-inlier-threshold-m", type=float, default=0.05)
    parser.add_argument("--synthetic-frame-count", type=int, default=12)
    parser.add_argument("--synthetic-width", type=int, default=640)
    parser.add_argument("--synthetic-height", type=int, default=384)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--real-depth", action="store_true")
    parser.add_argument("--real-pose", action="store_true")
    parser.add_argument("--mock-depth", action="store_true")
    parser.add_argument("--mock-pose", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for artifacts (default: benchmarks/results/reconstruct_compare_<timestamp>)",
    )
    args = parser.parse_args()

    backends = _parse_backends(args.backends)
    timestamp = utc_timestamp().replace("-", "").replace(":", "")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("benchmarks") / "results" / f"reconstruct_compare_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_rows: dict[str, dict[str, Any]] = {}
    for backend in backends:
        quality_path = output_dir / f"quality_{backend}.json"
        bench_path = output_dir / f"bench_{backend}.json"

        quality_cmd = [
            sys.executable,
            "benchmarks/quality_reconstruct.py",
            "--quality",
            args.quality,
            "--output-format",
            args.output_format,
            "--reconstruct-backend",
            backend,
            "--depth-backend",
            args.depth_backend,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--model-dir",
            args.model_dir,
            "--seed",
            str(args.seed),
            "--synthetic-frame-count",
            str(args.synthetic_frame_count),
            "--synthetic-width",
            str(args.synthetic_width),
            "--synthetic-height",
            str(args.synthetic_height),
            "--render-downscale",
            str(args.render_downscale),
            "--depth-inlier-threshold-m",
            str(args.depth_inlier_threshold_m),
            "--output",
            str(quality_path),
        ]
        if args.dataset_dir:
            quality_cmd.extend(["--dataset-dir", args.dataset_dir])
        if args.real_depth:
            quality_cmd.append("--real-depth")
        if args.real_pose:
            quality_cmd.append("--real-pose")

        bench_cmd = [
            sys.executable,
            "benchmarks/bench_reconstruct.py",
            "--quality",
            args.quality,
            "--output-format",
            args.output_format,
            "--reconstruct-backend",
            backend,
            "--backend",
            args.depth_backend,
            "--frame-counts",
            args.frame_counts,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--runs",
            str(args.runs),
            "--warmup",
            str(args.warmup),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--model-dir",
            args.model_dir,
            "--seed",
            str(args.seed),
            "--output",
            str(bench_path),
        ]
        if args.mock_depth:
            bench_cmd.append("--mock-depth")
        if args.mock_pose:
            bench_cmd.append("--mock-pose")

        _run_command(quality_cmd)
        _run_command(bench_cmd)

        quality_payload = _load_json(quality_path)
        bench_payload = _load_json(bench_path)
        if not args.allow_errors and _has_backend_failures(quality_payload, bench_payload):
            raise RuntimeError(
                f"Backend '{backend}' produced failed cases. Re-run with --allow-errors to emit partial report."
            )

        backend_rows[backend] = {
            "reconstruct_backend": backend,
            "quality_file": quality_path.name,
            "bench_file": bench_path.name,
            "quality_summary": _extract_quality_metrics(quality_payload),
            "bench_summary": _extract_bench_metrics(bench_payload),
            "quality_config": quality_payload.get("config", {}),
            "bench_config": bench_payload.get("config", {}),
        }

    baseline = backends[0]
    baseline_quality = backend_rows[baseline]["quality_summary"]
    baseline_bench = backend_rows[baseline]["bench_summary"]
    baseline_quality_config = backend_rows[baseline]["quality_config"]
    baseline_bench_config = backend_rows[baseline]["bench_config"]

    deltas: dict[str, dict[str, float | None]] = {}
    for backend in backends[1:]:
        quality = backend_rows[backend]["quality_summary"]
        bench = backend_rows[backend]["bench_summary"]
        deltas[backend] = {
            "coverage_mean": _delta_or_none(quality["coverage_mean"], baseline_quality["coverage_mean"]),
            "abs_depth_error_mean_m": _delta_or_none(
                quality["abs_depth_error_mean_m"],
                baseline_quality["abs_depth_error_mean_m"],
            ),
            "inlier_ratio_mean": _delta_or_none(quality["inlier_ratio_mean"], baseline_quality["inlier_ratio_mean"]),
            "latency_p50_ms_mean": _delta_or_none(
                bench["latency_p50_ms_mean"],
                baseline_bench["latency_p50_ms_mean"],
            ),
            "latency_p95_ms_mean": _delta_or_none(
                bench["latency_p95_ms_mean"],
                baseline_bench["latency_p95_ms_mean"],
            ),
            "throughput_fps_mean": _delta_or_none(
                bench["throughput_fps_mean"],
                baseline_bench["throughput_fps_mean"],
            ),
        }

    report_payload = {
        "schema_version": "reconstruct_compare.v1",
        "kind": "reconstruct_compare",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit_short(),
        "system": system_info(),
        "config": {
            "dataset_dir": args.dataset_dir,
            "depth_backend": args.depth_backend,
            "backends": backends,
            "quality": args.quality,
            "output_format": args.output_format,
            "frame_counts": args.frame_counts,
            "width": args.width,
            "height": args.height,
            "runs": args.runs,
            "warmup": args.warmup,
            "render_downscale": args.render_downscale,
            "depth_inlier_threshold_m": args.depth_inlier_threshold_m,
            "requested_device": args.device,
            "requested_dtype": args.dtype,
            "device": baseline_quality_config.get("device", baseline_bench_config.get("device", args.device)),
            "dtype": baseline_quality_config.get("dtype", baseline_bench_config.get("dtype", args.dtype)),
            "seed": args.seed,
            "real_depth": args.real_depth,
            "real_pose": args.real_pose,
            "mock_depth": args.mock_depth,
            "mock_pose": args.mock_pose,
        },
        "results": backend_rows,
        "deltas_vs_baseline": deltas,
    }

    report_json = output_dir / "reconstruct_compare.json"
    write_json(report_json, report_payload)

    report_md = output_dir / "reconstruct_compare.md"
    report_md.write_text(
        _build_markdown(report_payload=report_payload, backend_rows=backend_rows, deltas=deltas),
        encoding="utf-8",
    )

    print(f"Wrote compare JSON: {report_json}")
    print(f"Wrote compare Markdown: {report_md}")


if __name__ == "__main__":
    main()
