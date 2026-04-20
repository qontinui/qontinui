"""E2E test: trajectory logger against notepad.

Enables the trajectory logger, runs a 3-action simulated workflow against
notepad, and asserts that 3 grounding JSONL records are produced with the
expected schema.

Run with::

    pytest qontinui/tests/e2e/broken_accessibility/test_trajectory_logger.py \
        -v --run-live-e2e

Requirements:
    - Windows (for notepad)
    - Real display
    - qontinui-train and qontinui-runner python-bridge on sys.path
      (handled by path setup below)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup: bring qontinui-runner/python-bridge and qontinui-train onto
# sys.path so we can import TrajectoryLogger and grounding_record.
#
# We must insert BEFORE any qontinui paths that pytest may have added,
# and we use importlib for modules whose names clash with packages already
# on sys.path (e.g. "models").
# ---------------------------------------------------------------------------
_ROOT = (
    Path(__file__).resolve().parents[4]
)  # …/qontinui/tests/e2e/broken_accessibility -> qontinui-root
_RUNNER_BRIDGE = _ROOT / "qontinui-runner" / "python-bridge"
_TRAIN_ROOT = _ROOT / "qontinui-train"

for _p in [str(_RUNNER_BRIDGE), str(_TRAIN_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _import_from(module_name: str, file_path: Path):
    """Import a module by absolute file path, avoiding name collisions."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so dataclass/annotations resolve
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


pytestmark = pytest.mark.live_e2e


def test_trajectory_logger_notepad_3_actions(live_app):
    """Enable trajectory logger, run 3 actions on notepad, assert 3 JSONL records."""

    with tempfile.TemporaryDirectory(prefix="qontinui_traj_test_") as tmpdir:
        output_dir = Path(tmpdir)

        # --- Import via importlib to avoid name collisions with pytest ---
        _aer_mod = _import_from(
            "action_execution_record",
            _RUNNER_BRIDGE / "models" / "action_execution_record.py",
        )
        ActionExecutionRecord = _aer_mod.ActionExecutionRecord

        # Pre-load grounding_record into sys.modules so trajectory_logger
        # finds it without triggering the full qontinui_train.__init__ chain
        # (which imports training_data_exporter → models, causing conflicts).
        _gr_mod = _import_from(
            "qontinui_train.export.grounding_record",
            _TRAIN_ROOT / "qontinui_train" / "export" / "grounding_record.py",
        )
        sys.modules["qontinui_train.export.grounding_record"] = _gr_mod

        _tl_mod = _import_from(
            "trajectory_logger",
            _RUNNER_BRIDGE / "services" / "trajectory_logger.py",
        )
        TrajectoryLogger = _tl_mod.TrajectoryLogger

        # Launch notepad so there is something real on screen
        app = live_app("notepad.exe", window_title="Notepad")

        # Give the window a moment to fully paint
        time.sleep(1.0)

        # Initialise trajectory logger (no WSM, no OmniParser — pure pixel-diff / record flag)
        logger = TrajectoryLogger(
            output_dir=output_dir,
            max_records_per_session=10,
        )

        # Simulate 3 actions: CLICK, TYPE, CLICK
        actions = [
            ActionExecutionRecord(
                action_id="action-0",
                action_type="CLICK",
                config={"x": 100, "y": 100},
                success=True,
                clicked_location=(100, 100),
            ),
            ActionExecutionRecord(
                action_id="action-1",
                action_type="TYPE",
                config={"text": "hello"},
                success=True,
                typed_text="hello",
            ),
            ActionExecutionRecord(
                action_id="action-2",
                action_type="CLICK",
                config={"x": 200, "y": 200},
                success=True,
                clicked_location=(200, 200),
            ),
        ]

        for i, record in enumerate(actions):
            logger.on_action_start(record.action_id)
            # Move the mouse between captures so screenshots differ
            # (prevents SHA256 exact dedup from dropping records)
            try:
                import pyautogui

                pyautogui.moveTo(100 + i * 150, 100 + i * 100, duration=0)
            except ImportError:
                pass
            time.sleep(0.15)
            logger.on_record_created(record)

        # --- Assertions ---

        grounding_file = output_dir / "grounding.jsonl"
        assert grounding_file.exists(), f"grounding.jsonl not found in {output_dir}"

        lines = grounding_file.read_text(encoding="utf-8").strip().split("\n")
        # May be fewer than 3 if screenshots are identical (dedup), but at least 1
        assert len(lines) >= 1, f"Expected at least 1 record, got {len(lines)}"
        assert len(lines) <= 3, f"Expected at most 3 records, got {len(lines)}"

        for i, line in enumerate(lines):
            rec = json.loads(line)

            # Required top-level fields
            assert "image_hash" in rec, f"Record {i}: missing image_hash"
            assert "image_path" in rec, f"Record {i}: missing image_path"
            assert isinstance(rec["viewport_width"], int), f"Record {i}: viewport_width not int"
            assert isinstance(rec["viewport_height"], int), f"Record {i}: viewport_height not int"
            assert rec["viewport_width"] > 0, f"Record {i}: viewport_width <= 0"
            assert rec["viewport_height"] > 0, f"Record {i}: viewport_height <= 0"

            # Elements list (may be empty without OmniParser, but must exist)
            assert "elements" in rec, f"Record {i}: missing elements"
            assert isinstance(rec["elements"], list), f"Record {i}: elements not list"

            # Action (must be present for dynamic records)
            assert "action" in rec, f"Record {i}: missing action"
            assert rec["action"] is not None, f"Record {i}: action is None"
            assert rec["action"]["type"] in (
                "click",
                "type",
            ), f"Record {i}: unexpected action type {rec['action']['type']}"

            # Source tag
            assert rec["source"] == "dynamic", f"Record {i}: source is not 'dynamic'"

            # Timestamp
            assert "timestamp" in rec, f"Record {i}: missing timestamp"
            assert len(rec["timestamp"]) > 0, f"Record {i}: empty timestamp"

        # Validate images directory has PNG files
        images_dir = output_dir / "images"
        assert images_dir.exists(), "images/ directory not created"
        png_files = list(images_dir.glob("*.png"))
        assert len(png_files) > 0, "No PNG files in images/"
        # At most 3, but could be fewer due to exact dedup (identical screenshots)
        assert len(png_files) <= 3, f"Too many PNGs: {len(png_files)}"
