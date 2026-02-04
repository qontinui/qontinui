"""Snapshot recorder for capturing automation execution.

Records automation runs in REAL mode for playback in SCREENSHOT/MOCK modes.
Saves screenshots, matches, and action results to disk in a structured format.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qontinui_schemas.common import utc_now

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

from qontinui.mock.snapshot import ActionHistory, ActionRecord
from qontinui.model.match import Match

logger = logging.getLogger(__name__)


@dataclass
class RecorderConfig:
    """Configuration for snapshot recorder."""

    base_dir: Path
    """Base directory for all snapshot runs."""

    run_id: str | None = None
    """Run identifier. Auto-generated if None."""

    save_screenshots: bool = True
    """Whether to save full screenshots."""

    save_patterns: bool = True
    """Whether to save pattern histories."""

    save_action_log: bool = True
    """Whether to save chronological action log."""

    compression: str = "png"
    """Image compression format: 'png' or 'jpg'."""

    compression_quality: int = 95
    """JPEG quality (1-100) if compression='jpg'."""


class SnapshotRecorder:
    """Records automation execution for playback in screenshot/mock modes.

    Usage:
        config = RecorderConfig(base_dir=Path("/snapshots"))
        recorder = SnapshotRecorder(config)

        # During automation
        recorder.record_find_action(pattern_id, pattern_name, matches, screenshot, states, duration)
        recorder.record_screenshot(screenshot)

        # When done
        recorder.finalize()
        snapshot_dir = recorder.get_snapshot_directory()
    """

    def __init__(self, config: RecorderConfig) -> None:
        """Initialize recorder.

        Args:
            config: Recorder configuration
        """
        self.config = config

        # Generate run ID if not provided
        if config.run_id:
            self.run_id = config.run_id
        else:
            self.run_id = f"run-{utc_now().strftime('%Y-%m-%d-%H%M%S')}"

        # Create run directory
        if isinstance(config.base_dir, str):
            self.run_dir: Path = Path(config.base_dir) / self.run_id
        else:
            self.run_dir = config.base_dir / self.run_id

        # Counters
        self.action_count = 0
        self.screenshot_count = 0
        self.first_screenshot_recorded = False

        # Storage
        self.action_log: list[dict[str, Any]] = []
        self.pattern_histories: dict[str, ActionHistory] = {}

        # Metadata
        self.start_time = utc_now()
        self.end_time: datetime | None = None

        # Create directory structure
        self._create_directories()

        logger.info(f"SnapshotRecorder initialized: {self.run_dir}")

    def _create_directories(self):
        """Create snapshot directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "screenshots").mkdir(exist_ok=True)
        (self.run_dir / "patterns").mkdir(exist_ok=True)
        (self.run_dir / "regions").mkdir(exist_ok=True)

    def record_find_action(
        self,
        pattern_id: str,
        pattern_name: str,
        matches: list[Match],
        screenshot: Any | None,  # Image.Image type hint causes issues if PIL not installed
        active_states: set[str],
        duration_ms: float,
    ):
        """Record a FIND action with its results.

        Args:
            pattern_id: Unique pattern identifier
            pattern_name: Human-readable pattern name
            matches: List of Match objects found
            screenshot: Screenshot image (PIL Image) or None
            active_states: Set of active state IDs
            duration_ms: Search duration in milliseconds
        """
        self.action_count += 1

        # Save screenshot if provided
        screenshot_file = None
        if screenshot and self.config.save_screenshots:
            try:
                screenshot_file = self._save_screenshot(screenshot)
            except Exception as e:
                logger.error(f"Failed to save screenshot: {e}")

        # Create ActionRecord
        record = ActionRecord(
            action_type="find",
            action_success=len(matches) > 0,
            match_list=matches,  # Uses actual Match objects
            duration=duration_ms / 1000.0,  # Convert ms to seconds
            timestamp=utc_now(),
            active_states=active_states,
            metadata={
                "pattern_id": pattern_id,
                "pattern_name": pattern_name,
                "screenshot_file": screenshot_file,
                "matches_count": len(matches),
            },
        )

        # Add to pattern history
        if self.config.save_patterns:
            if pattern_id not in self.pattern_histories:
                self.pattern_histories[pattern_id] = ActionHistory()

            self.pattern_histories[pattern_id].add_record(record)

        # Add to action log
        if self.config.save_action_log:
            # Mark first screenshot as start screenshot
            is_start_screenshot = False
            if screenshot_file and not self.first_screenshot_recorded:
                is_start_screenshot = True
                self.first_screenshot_recorded = True

            self.action_log.append(
                {
                    "action_id": self.action_count,
                    "action_type": "find",
                    "timestamp": utc_now().isoformat(),
                    "pattern_id": pattern_id,
                    "pattern_name": pattern_name,
                    "success": record.action_success,
                    "duration_ms": duration_ms,
                    "screenshot_file": screenshot_file,
                    "matches_count": len(matches),
                    "active_states": list(active_states),
                    "is_start_screenshot": is_start_screenshot,
                }
            )

        logger.debug(
            f"Recorded FIND action: {pattern_name} ({len(matches)} matches, {duration_ms:.1f}ms)"
        )

    def record_screenshot(self, screenshot: Any) -> str:
        """Record a screenshot and return its filename.

        Args:
            screenshot: Screenshot image (PIL Image)

        Returns:
            Relative path to saved screenshot (e.g., "screenshots/screenshot-001.png")
        """
        if not self.config.save_screenshots:
            return ""

        try:
            return self._save_screenshot(screenshot)
        except Exception as e:
            logger.error(f"Failed to record screenshot: {e}")
            return ""

    def _save_screenshot(self, screenshot: Any) -> str:
        """Save screenshot to disk and return filename.

        Args:
            screenshot: PIL Image object

        Returns:
            Relative path to screenshot file
        """
        if Image is None:
            raise ImportError("PIL/Pillow is required for screenshot recording")

        self.screenshot_count += 1
        filename = f"screenshot-{self.screenshot_count:03d}.{self.config.compression}"
        filepath = self.run_dir / "screenshots" / filename

        # Save with appropriate format
        if self.config.compression == "jpg":
            screenshot.save(
                str(filepath),
                format="JPEG",
                quality=self.config.compression_quality,
                optimize=True,
            )
        else:
            screenshot.save(str(filepath), format="PNG", optimize=True)

        # Return relative path
        return f"screenshots/{filename}"

    def save_pattern_history(self, pattern_id: str):
        """Save ActionHistory for a pattern to disk.

        Args:
            pattern_id: Pattern identifier
        """
        if pattern_id not in self.pattern_histories:
            logger.warning(f"No history for pattern: {pattern_id}")
            return

        history = self.pattern_histories[pattern_id]
        pattern_dir = self.run_dir / "patterns" / pattern_id
        pattern_dir.mkdir(parents=True, exist_ok=True)

        # Save history.json
        history_file = pattern_dir / "history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(history.to_dict(), f, indent=2)  # type: ignore[attr-defined]
            logger.debug(f"Saved pattern history: {pattern_id}")
        except Exception as e:
            logger.error(f"Failed to save pattern history {pattern_id}: {e}")

        # Save individual match files
        for i, record in enumerate(history.records):  # type: ignore[attr-defined]
            for j, match in enumerate(record.match_list):
                match_file = pattern_dir / f"match-{i:03d}-{j:02d}.json"
                try:
                    with open(match_file, "w") as f:
                        json.dump(self._match_to_dict(match), f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save match {match_file}: {e}")

    def _match_to_dict(self, match: Match) -> dict[str, Any]:
        """Convert Match to dictionary for JSON serialization.

        Args:
            match: Match object

        Returns:
            Dictionary representation
        """
        return {
            "x": match.x,
            "y": match.y,
            "width": match.w,
            "height": match.h,
            "score": match.score,
            "region": (
                {
                    "x": match.region.x,
                    "y": match.region.y,
                    "width": match.region.w,
                    "height": match.region.h,
                }
                if match.region
                else None
            ),
            "search_area": (
                {
                    "x": match.search_area.x,
                    "y": match.search_area.y,
                    "width": match.search_area.w,
                    "height": match.search_area.h,
                }
                if match.search_area
                else None
            ),
            "image_name": match.image.name if match.image else None,
        }

    def finalize(self):
        """Finalize recording and save all data to disk."""
        self.end_time = utc_now()

        logger.info(f"Finalizing snapshot recording: {self.run_id}")

        # Save all pattern histories
        if self.config.save_patterns:
            for pattern_id in self.pattern_histories:
                self.save_pattern_history(pattern_id)

        # Save action log
        if self.config.save_action_log:
            action_log_file = self.run_dir / "action_log.json"
            try:
                with open(action_log_file, "w") as f:
                    json.dump({"actions": self.action_log}, f, indent=2)
                logger.debug("Saved action log")
            except Exception as e:
                logger.error(f"Failed to save action log: {e}")

        # Save metadata
        self._save_metadata()

        logger.info(
            f"Snapshot recording complete: {self.run_id} "
            f"({self.action_count} actions, {self.screenshot_count} screenshots)"
        )

    def _save_metadata(self):
        """Save run metadata to disk."""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else None

        metadata = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "total_actions": self.action_count,
            "total_screenshots": self.screenshot_count,
            "patterns_recorded": len(self.pattern_histories),
            "pattern_ids": list(self.pattern_histories.keys()),
            "config": {
                "save_screenshots": self.config.save_screenshots,
                "save_patterns": self.config.save_patterns,
                "compression": self.config.compression,
            },
        }

        metadata_file = self.run_dir / "metadata.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug("Saved metadata")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_snapshot_directory(self) -> Path:
        """Get the directory where snapshots are being saved.

        Returns:
            Path to snapshot directory
        """
        return self.run_dir  # type: ignore[no-any-return]

    def get_statistics(self) -> dict[str, Any]:
        """Get current recording statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_dir),
            "actions_recorded": self.action_count,
            "screenshots_saved": self.screenshot_count,
            "patterns_tracked": len(self.pattern_histories),
            "elapsed_seconds": (utc_now() - self.start_time).total_seconds(),
        }
