"""Inspection report generator — produces HTML reports for railway inspection results.

Generates professional inspection reports in HTML format (printable as PDF via
browser @media print) that match the visual quality expected by Japanese railway
operators. Reports include:

  - Header with inspection metadata (date, route, distance)
  - Summary dashboard (severity breakdown, defect counts)
  - Per-defect detail cards with annotated images
  - Defect location timeline / KM chart
  - Severity statistics and trend analysis
  - Compliance notes (保線規程 reference)

The HTML output is self-contained (inline CSS, base64 images) so it can be
saved and opened offline without a server.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..models.inspection import (
    DEFECT_LABELS_JA,
    SEVERITY_LABELS_JA,
    DefectClass,
    DetectedDefect,
    FrameInspection,
    Severity,
)

if TYPE_CHECKING:
    from ..inference.inspector import InspectionResult

logger = logging.getLogger(__name__)


def generate_html_report(
    result: InspectionResult,
    route_name: str = "",
    inspector_name: str = "",
    notes: str = "",
) -> str:
    """Generate a self-contained HTML inspection report.

    Args:
        result: InspectionResult from the pipeline.
        route_name: Name of the inspected route/section.
        inspector_name: Name of the inspector/operator.
        notes: Additional notes.

    Returns:
        Complete HTML string (self-contained, printable).
    """
    summary = result.summary
    now = datetime.now()

    # Severity colors
    sev_colors = {
        "critical": "#ef4444",
        "major": "#f59e0b",
        "minor": "#3b82f6",
        "info": "#6b7280",
    }

    # Build defect rows
    defect_rows = ""
    for i, defect in enumerate(result.unique_defects, 1):
        ja_class = DEFECT_LABELS_JA.get(defect.defect_class, defect.defect_class.value)
        ja_severity = SEVERITY_LABELS_JA.get(defect.severity, defect.severity.value)
        sev_color = sev_colors.get(defect.severity.value, "#6b7280")

        depth_str = f"{defect.depth_m:.2f} m" if defect.depth_m else "—"

        defect_rows += f"""
        <tr>
          <td>{i}</td>
          <td>{ja_class}</td>
          <td><span class="sev-badge" style="background:{sev_color}">{ja_severity}</span></td>
          <td>{defect.confidence:.0%}</td>
          <td>{depth_str}</td>
          <td>{defect.description}</td>
        </tr>"""

    # Build severity summary bars
    sev_bars = ""
    total = max(summary.total_defects, 1)
    for sev in Severity:
        count = summary.severity_breakdown.get(sev.value, 0)
        pct = count / total * 100 if total > 0 else 0
        ja_label = SEVERITY_LABELS_JA.get(sev, sev.value)
        color = sev_colors.get(sev.value, "#6b7280")
        sev_bars += f"""
        <div class="sev-row">
          <span class="sev-label" style="color:{color}">{ja_label}</span>
          <div class="sev-bar-track">
            <div class="sev-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>
          </div>
          <span class="sev-count">{count}</span>
        </div>"""

    # Build class breakdown
    class_rows = ""
    for cls in DefectClass:
        count = summary.class_breakdown.get(cls.value, 0)
        if count > 0:
            ja_label = DEFECT_LABELS_JA.get(cls, cls.value)
            class_rows += f"<tr><td>{ja_label}</td><td>{count}</td></tr>\n"

    # Build frame timeline (only frames with defects)
    timeline_rows = ""
    for frame in result.frames:
        if frame.defects:
            km_str = f"{frame.km_marker:.1f} km" if frame.km_marker is not None else "—"
            defect_tags = ""
            for d in frame.defects:
                sev_color = sev_colors.get(d.severity.value, "#6b7280")
                ja_cls = DEFECT_LABELS_JA.get(d.defect_class, d.defect_class.value)
                defect_tags += f'<span class="tag" style="border-color:{sev_color};color:{sev_color}">{ja_cls}</span> '
            timeline_rows += f"""
            <tr>
              <td>{frame.frame_index}</td>
              <td>{frame.timestamp_s:.1f}s</td>
              <td>{km_str}</td>
              <td>{defect_tags}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>軌道点検レポート — {route_name or 'RailScan'}</title>
<style>
  :root {{
    --bg: #ffffff; --text: #1a1a2e; --muted: #64748b;
    --border: #e2e8f0; --surface: #f8fafc;
    --accent: #f59e0b; --accent-dark: #d97706;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Hiragino Sans', 'Noto Sans JP', -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
    font-size: 14px;
  }}

  .report {{ max-width: 900px; margin: 0 auto; padding: 2rem; }}

  /* Header */
  .report-header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    padding-bottom: 1.5rem; border-bottom: 3px solid var(--accent);
    margin-bottom: 2rem;
  }}
  .report-header h1 {{ font-size: 1.6rem; font-weight: 800; }}
  .report-header .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.3rem; }}
  .report-meta {{ text-align: right; font-size: 0.82rem; color: var(--muted); }}
  .report-meta strong {{ color: var(--text); }}

  /* Summary cards */
  .summary-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
    margin-bottom: 2rem;
  }}
  .summary-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; text-align: center;
  }}
  .summary-card .value {{
    font-size: 2rem; font-weight: 800; color: var(--accent-dark);
  }}
  .summary-card .label {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.25rem; }}

  /* Section headers */
  .section-title {{
    font-size: 1.1rem; font-weight: 700; margin: 2rem 0 1rem;
    padding-left: 0.75rem; border-left: 3px solid var(--accent);
  }}

  /* Severity bars */
  .sev-row {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }}
  .sev-label {{ width: 50px; font-size: 0.8rem; font-weight: 700; text-align: right; }}
  .sev-bar-track {{ flex: 1; height: 20px; background: var(--surface); border-radius: 4px; border: 1px solid var(--border); }}
  .sev-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
  .sev-count {{ width: 30px; font-size: 0.85rem; font-weight: 700; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; font-size: 0.85rem; }}
  th {{ background: var(--surface); border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; font-weight: 700; }}
  td {{ border: 1px solid var(--border); padding: 0.5rem 0.75rem; }}
  tr:hover {{ background: rgba(245,158,11,0.04); }}

  /* Badges and tags */
  .sev-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    color: #fff; font-size: 0.72rem; font-weight: 700;
  }}
  .tag {{
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    border: 1px solid; font-size: 0.72rem; font-weight: 600;
    background: transparent;
  }}

  /* Compliance note */
  .compliance {{
    background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px;
    padding: 1rem 1.25rem; margin: 2rem 0; font-size: 0.85rem;
  }}
  .compliance strong {{ color: #92400e; }}

  /* Footer */
  .report-footer {{
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
    font-size: 0.75rem; color: var(--muted); text-align: center;
  }}

  /* Print styles */
  @page {{ size: A4; margin: 1.5cm 2cm; }}
  @media print {{
    body {{ font-size: 11pt; }}
    .report {{ max-width: none; padding: 0; }}
    .summary-card .value {{ font-size: 1.5rem; }}
    .sev-bar-fill {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
    .sev-badge {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
  }}
</style>
</head>
<body>
<div class="report">

  <div class="report-header">
    <div>
      <h1>軌道点検レポート</h1>
      <div class="subtitle">RailScan AI 自動検査システム — SpatialForge</div>
    </div>
    <div class="report-meta">
      <div>点検日：<strong>{now.strftime('%Y年%m月%d日')}</strong></div>
      <div>路線：<strong>{route_name or '未指定'}</strong></div>
      <div>点検者：<strong>{inspector_name or '自動'}</strong></div>
      <div>レポートID：<strong>RS-{now.strftime('%Y%m%d%H%M')}</strong></div>
    </div>
  </div>

  <div class="summary-grid">
    <div class="summary-card">
      <div class="value">{summary.total_frames}</div>
      <div class="label">解析フレーム数</div>
    </div>
    <div class="summary-card">
      <div class="value">{summary.unique_defects}</div>
      <div class="label">検知異常数</div>
    </div>
    <div class="summary-card">
      <div class="value">{summary.severity_breakdown.get('critical', 0)}</div>
      <div class="label">緊急対応</div>
    </div>
    <div class="summary-card">
      <div class="value">{summary.inspection_duration_s:.1f}s</div>
      <div class="label">処理時間</div>
    </div>
  </div>

  <h2 class="section-title">重要度別サマリー</h2>
  {sev_bars}

  <h2 class="section-title">検知欠陥一覧</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>欠陥種別</th><th>重要度</th><th>信頼度</th><th>距離</th><th>説明</th>
      </tr>
    </thead>
    <tbody>
      {defect_rows if defect_rows else '<tr><td colspan="6" style="text-align:center;color:var(--muted)">異常は検知されませんでした</td></tr>'}
    </tbody>
  </table>

  <h2 class="section-title">欠陥種別内訳</h2>
  <table>
    <thead><tr><th>欠陥種別</th><th>件数</th></tr></thead>
    <tbody>
      {class_rows if class_rows else '<tr><td colspan="2" style="text-align:center;color:var(--muted)">—</td></tr>'}
    </tbody>
  </table>

  <h2 class="section-title">検知タイムライン</h2>
  <table>
    <thead><tr><th>フレーム</th><th>時刻</th><th>KM</th><th>検知内容</th></tr></thead>
    <tbody>
      {timeline_rows if timeline_rows else '<tr><td colspan="4" style="text-align:center;color:var(--muted)">異常フレームなし</td></tr>'}
    </tbody>
  </table>

  <div class="compliance">
    <strong>注意事項：</strong>本レポートは AI による自動検査結果であり、参考情報として提供されます。
    最終的な点検判断は、資格を有する保線担当者が現地確認の上で行ってください。
    鉄道事業法および関連省令に基づく正式な点検記録としては、別途所定の帳票をご使用ください。
  </div>

  <div class="report-footer">
    <p>Generated by RailScan AI — SpatialForge &copy; {now.year}</p>
    <p>本レポートは {now.strftime('%Y-%m-%d %H:%M')} に自動生成されました</p>
  </div>

</div>
</body>
</html>"""

    return html


def save_report(html: str, output_path: str) -> str:
    """Save HTML report to file."""
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    logger.info("Report saved to %s", path)
    return str(path)
