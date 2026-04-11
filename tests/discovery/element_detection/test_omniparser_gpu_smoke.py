"""GPU smoke test — verifies the host-side OmniParserDetector actually
uses CUDA when torch sees a GPU.

Opt-in via ``--run-gpu-smoke``. Skipped in CI / on CPU-only machines.
"""

from __future__ import annotations

import numpy as np
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover
    # Hook only fires if this file is treated as a conftest; harmless otherwise.
    parser.addoption(
        "--run-gpu-smoke",
        action="store_true",
        default=False,
        help="Run GPU smoke tests that require a real CUDA device.",
    )


@pytest.fixture
def _skip_if_no_gpu() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.mark.gpu_smoke
def test_omniparser_yolo_runs_on_gpu(_skip_if_no_gpu: None) -> None:
    from qontinui.discovery.element_detection.omniparser_detector import (
        OmniParserDetector,
    )
    from qontinui.find.backends.omniparser_config import OmniParserSettings

    d = OmniParserDetector(settings=OmniParserSettings(enabled=True, device="cuda"))
    d._ensure_yolo_loaded()
    assert d._device == "cuda"
    assert d._yolo_model is not None
    params = list(d._yolo_model.model.parameters())
    assert params and params[0].is_cuda, "YOLO params should live on CUDA"

    # Empty image — YOLO may or may not detect things, but the call must
    # succeed and not fall back to CPU mid-run.
    haystack = np.zeros((1080, 1920, 3), dtype=np.uint8)
    regions = d.get_interactive_regions(haystack)
    assert isinstance(regions, list)
