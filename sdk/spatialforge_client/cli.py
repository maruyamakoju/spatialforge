"""SpatialForge CLI — command-line interface for the SpatialForge API.

Usage:
    spatialforge depth photo.jpg
    spatialforge measure room.jpg --p1 100,200 --p2 500,200
    spatialforge pose video.mp4
    spatialforge reconstruct walkthrough.mp4 --quality high
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

API_KEY_ENV = "SPATIALFORGE_API_KEY"
BASE_URL_ENV = "SPATIALFORGE_BASE_URL"


def _get_client(**kwargs):
    """Create a SpatialForge client from env vars or options."""
    from .client import Client

    api_key = kwargs.get("api_key") or os.environ.get(API_KEY_ENV)
    if not api_key:
        click.echo(
            f"Error: API key required. Set {API_KEY_ENV} or use --api-key.",
            err=True,
        )
        sys.exit(1)

    base_url = kwargs.get("base_url") or os.environ.get(BASE_URL_ENV)
    kw = {"api_key": api_key}
    if base_url:
        kw["base_url"] = base_url

    return Client(**kw)


def _format_time(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def _print_json(data: dict) -> None:
    """Pretty-print JSON output."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()
        text = json.dumps(data, indent=2, default=str)
        console.print(Syntax(text, "json", theme="monokai"))
    except ImportError:
        click.echo(json.dumps(data, indent=2, default=str))


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a key-value table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, show_header=False, border_style="dim")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="white")
        for k, v in rows:
            table.add_row(k, str(v))
        console.print(table)
    except ImportError:
        click.echo(f"\n  {title}")
        click.echo("  " + "─" * 40)
        for k, v in rows:
            click.echo(f"  {k:20s} {v}")
        click.echo()


@click.group()
@click.version_option(package_name="spatialforge-client")
def main():
    """SpatialForge CLI — spatial intelligence from any camera."""
    pass


@main.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--model", "-m", default="large", help="Model: giant, large, base, small")
@click.option("--format", "-f", "output_format", default="png16", help="Output: png16, exr, npy")
@click.option("--no-metric", is_flag=True, help="Output relative depth instead of metric")
@click.option("--output", "-o", type=click.Path(), help="Save depth map to file")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", envvar=API_KEY_ENV, help="API key")
@click.option("--base-url", envvar=BASE_URL_ENV, help="API base URL")
def depth(image, model, output_format, no_metric, output, as_json, api_key, base_url):
    """Estimate depth from an image."""
    client = _get_client(api_key=api_key, base_url=base_url)
    try:
        result = client.depth(
            image, model=model, output_format=output_format, metric=not no_metric
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    if as_json:
        _print_json(result._raw)
        return

    _print_table("Depth Estimation", [
        ("Image", Path(image).name),
        ("Model", model),
        ("Resolution", f"{result.width} x {result.height}"),
        ("Min depth", f"{result.min_depth_m:.3f} m"),
        ("Max depth", f"{result.max_depth_m:.3f} m"),
        ("Focal length", f"{result.focal_length_px:.1f} px" if result.focal_length_px else "N/A"),
        ("Confidence", f"{result.confidence_mean:.1%}"),
        ("Processing", _format_time(result.processing_time_ms)),
        ("Depth map", result.depth_map_url),
    ])

    if output:
        click.echo(f"Saving depth map to {output}...")
        result.save_depth_map(output)
        click.echo("Done.")


@main.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--p1", required=True, help="First point: x,y")
@click.option("--p2", required=True, help="Second point: x,y")
@click.option("--reference", "-r", help="Reference object JSON")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", envvar=API_KEY_ENV, help="API key")
@click.option("--base-url", envvar=BASE_URL_ENV, help="API base URL")
def measure(image, p1, p2, reference, as_json, api_key, base_url):
    """Measure distance between two points in an image."""
    try:
        x1, y1 = [float(v) for v in p1.split(",")]
        x2, y2 = [float(v) for v in p2.split(",")]
    except ValueError:
        click.echo("Error: Points must be in x,y format (e.g., --p1 100,200)", err=True)
        sys.exit(1)

    ref_obj = None
    if reference:
        ref_obj = json.loads(reference)

    client = _get_client(api_key=api_key, base_url=base_url)
    try:
        result = client.measure(image, (x1, y1), (x2, y2), reference_object=ref_obj)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    if as_json:
        _print_json({
            "distance_m": result.distance_m,
            "distance_cm": result.distance_cm,
            "confidence": result.confidence,
            "calibration_method": result.calibration_method,
        })
        return

    _print_table("Distance Measurement", [
        ("Image", Path(image).name),
        ("Point 1", f"({x1:.0f}, {y1:.0f})"),
        ("Point 2", f"({x2:.0f}, {y2:.0f})"),
        ("Distance", f"{result.distance_m:.3f} m ({result.distance_cm:.1f} cm)"),
        ("Confidence", f"{result.confidence:.1%}"),
        ("Calibration", result.calibration_method),
        ("Processing", _format_time(result.processing_time_ms)),
    ])


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--images", "-i", is_flag=True, help="Input is images (glob pattern)")
@click.option("--pointcloud", is_flag=True, help="Output sparse point cloud")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", envvar=API_KEY_ENV, help="API key")
@click.option("--base-url", envvar=BASE_URL_ENV, help="API base URL")
def pose(input_file, images, pointcloud, as_json, api_key, base_url):
    """Estimate camera poses from video or images."""
    client = _get_client(api_key=api_key, base_url=base_url)

    try:
        if images:
            import glob as g
            files = sorted(g.glob(str(input_file)))
            if not files:
                click.echo(f"No files matching {input_file}", err=True)
                sys.exit(1)
            result = client.pose(images=files, output_pointcloud=pointcloud)
        else:
            result = client.pose(video=input_file, output_pointcloud=pointcloud)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    if as_json:
        poses_data = []
        for p in result.camera_poses:
            poses_data.append({
                "frame": p.frame_index,
                "rotation": p.rotation,
                "translation": p.translation,
                "intrinsics": {"fx": p.fx, "fy": p.fy, "cx": p.cx, "cy": p.cy},
            })
        _print_json({"num_frames": result.num_frames, "poses": poses_data})
        return

    _print_table("Camera Pose Estimation", [
        ("Input", Path(input_file).name),
        ("Frames", str(result.num_frames)),
        ("Point cloud", result.pointcloud_url or "N/A"),
        ("Processing", _format_time(result.processing_time_ms)),
    ])

    for p in result.camera_poses[:5]:
        t = p.translation
        click.echo(f"  Frame {p.frame_index}: T=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    if len(result.camera_poses) > 5:
        click.echo(f"  ... and {len(result.camera_poses) - 5} more frames")


@main.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--quality", "-q", default="standard", help="Quality: draft, standard, high")
@click.option("--output-type", "-t", default="gaussian", help="Output: gaussian, pointcloud, mesh")
@click.option("--wait/--no-wait", default=True, help="Wait for completion")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", envvar=API_KEY_ENV, help="API key")
@click.option("--base-url", envvar=BASE_URL_ENV, help="API base URL")
def reconstruct(video, quality, output_type, wait, as_json, api_key, base_url):
    """Start 3D reconstruction from video."""
    client = _get_client(api_key=api_key, base_url=base_url)

    try:
        job = client.reconstruct(video, quality=quality, output=output_type)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Job {job.job_id} submitted (status: {job.status})")
    if job.estimated_time_s:
        click.echo(f"Estimated time: {job.estimated_time_s:.0f}s")

    if not wait:
        client.close()
        return

    try:
        click.echo("Waiting for completion...")
        result = job.wait(poll_interval=5.0)
    except (RuntimeError, TimeoutError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    if as_json:
        _print_json(result)
    else:
        click.echo(f"Complete! Scene URL: {result.get('scene_url', 'N/A')}")


@main.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--format", "-f", "output_format", default="svg", help="Output: svg, dxf, json")
@click.option("--wait/--no-wait", default=True, help="Wait for completion")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", envvar=API_KEY_ENV, help="API key")
@click.option("--base-url", envvar=BASE_URL_ENV, help="API base URL")
def floorplan(video, output_format, wait, as_json, api_key, base_url):
    """Generate floor plan from walkthrough video."""
    client = _get_client(api_key=api_key, base_url=base_url)

    try:
        job = client.floorplan(video, output_format=output_format)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Job {job.job_id} submitted (status: {job.status})")

    if not wait:
        client.close()
        return

    try:
        click.echo("Waiting for completion...")
        result = job.wait(poll_interval=5.0)
    except (RuntimeError, TimeoutError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    if as_json:
        _print_json(result)
    else:
        click.echo(f"Complete! Floor area: {result.get('floor_area_m2', 'N/A')} m2")


@main.command("segment-3d")
@click.argument("video", type=click.Path(exists=True))
@click.option("--prompt", "-p", required=True, help="What to segment (e.g., 'chairs')")
@click.option("--wait/--no-wait", default=True, help="Wait for completion")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", envvar=API_KEY_ENV, help="API key")
@click.option("--base-url", envvar=BASE_URL_ENV, help="API base URL")
def segment_3d(video, prompt, wait, as_json, api_key, base_url):
    """Segment 3D objects in video with text prompt."""
    client = _get_client(api_key=api_key, base_url=base_url)

    try:
        job = client.segment_3d(video, prompt=prompt)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Job {job.job_id} submitted (status: {job.status})")

    if not wait:
        client.close()
        return

    try:
        click.echo("Waiting for completion...")
        result = job.wait(poll_interval=5.0)
    except (RuntimeError, TimeoutError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    if as_json:
        _print_json(result)
    else:
        click.echo(f"Complete! Segments: {result.get('num_segments', 'N/A')}")


if __name__ == "__main__":
    main()
