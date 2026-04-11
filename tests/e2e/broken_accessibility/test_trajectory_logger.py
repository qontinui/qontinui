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

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup: bring qontinui-runner/python-bridge and qontinui-train onto
# sys.path so we can import TrajectoryLogger and grounding_record.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[4]  # qontinui-root/qontinui -> qontinui-root
_RUNNER_BRIDGE = _ROOT / "qontinui-runner" / "python-bridge"
_TRAIN_ROOT = _ROOT / "qontinui-train"

for _p in [str(_RUNNER_BRIDGE), str(_TRAIN_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.live_e2e


def test_trajectory_logger_notepad_3_actions(live_app):
    """Enable trajectory logger, run 3 actions on notepad, assert 3 JSONL records."""

    with tempfile.TemporaryDirectory(prefix="qontinui_traj_test_") as tmpdir:
        output_dir = Path(tmpdir)

        # --- Import after path setup ---
        from models.action_execution_record import ActionExecutionRecord
        from services.trajectory_logger import TrajectoryLogger

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

        for record in actions:
            logger.on_action_start(record.action_id)
            # Small delay to ensure post-screenshot differs from pre
            time.sleep(0.05)
            logger.on_record_created(record)

        # --- Assertions ---

        grounding_file = output_dir / "grounding.jsonl"
        assert grounding_file.exists(), f"grounding.jsonl not found in {output_dir}"

        lines = grounding_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3, f"Expected 3 records, got {len(lines)}"

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
            assert rec["action"]["type"] in ("click", "type"), (
                f"Record {i}: unexpected action type {rec['action']['type']}"
            )

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
