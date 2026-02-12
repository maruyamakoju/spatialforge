"""DepthEngine behavior tests for DA3-specific metric scaling."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from spatialforge.inference.depth_engine import DepthEngine
from spatialforge.inference.model_manager import ModelInfo


def test_da3_metric_depth_uses_focal_scaling_formula():
    mm = MagicMock()
    pipe = MagicMock(return_value={"depth": np.full((4, 4), 2.0, dtype=np.float32)})
    info = ModelInfo(
        repo="depth-anything/DA3METRIC-LARGE",
        license="apache-2.0",
        task="metric_depth",
        description="test",
    )
    mm.get_depth_model.return_value = (pipe, info)

    engine = DepthEngine(mm)
    image_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    result = engine.estimate(image_rgb=image_rgb, focal_length_px=600.0)

    assert result.is_metric is True
    assert np.allclose(result.depth_map, 4.0)
