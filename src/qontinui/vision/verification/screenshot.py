"""Screenshot management for vision verification.

Provides screenshot capture, caching, annotation, and storage
for assertion results and debugging.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators.base import LocatorMatch

logger = logging.getLogger(__name__)


class ScreenshotManager:
    """Manages screenshots for vision verification.

    Handles:
    - Screenshot capture (via HAL or direct)
    - Screenshot caching for repeated assertions
    - Failure screenshot annotation
    - Screenshot storage management
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        save_directory: str | Path | None = None,
    ) -> None:
        """Initialize screenshot manager.

        Args:
            config: Vision configuration.
            save_directory: Directory for saving screenshots.
        """
        self._config = config
        self._save_directory = Path(save_directory or ".dev-logs/screenshots")

        # Screenshot cache
        self._cache: NDArray[np.uint8] | None = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 0.5  # Cache TTL in seconds

        # HAL reference for capture
        self._hal: Any = None

    def set_hal(self, hal: Any) -> None:
        """Set HAL reference for screenshot capture.

        Args:
            hal: HAL instance with capture_screenshot method.
        """
        self._hal = hal

    def set_cache_ttl(self, ttl_seconds: float) -> None:
        """Set cache TTL.

        Args:
            ttl_seconds: Cache time-to-live in seconds.
        """
        self._cache_ttl = ttl_seconds

    def invalidate_cache(self) -> None:
        """Invalidate cached screenshot."""
        self._cache = None
        self._cache_timestamp = 0.0

    async def capture(self, force: bool = False) -> NDArray[np.uint8]:
        """Capture a screenshot.

        Args:
            force: Force new capture, ignoring cache.

        Returns:
            Screenshot as numpy array (BGR format).
        """
        # Check cache
        if not force and self._cache is not None:
            if time.time() - self._cache_timestamp < self._cache_ttl:
                return self._cache

        # Capture via HAL
        if self._hal is not None:
            screenshot = await self._capture_via_hal()
        else:
            # Fallback to direct capture
            screenshot = self._capture_direct()

        # Update cache
        self._cache = screenshot
        self._cache_timestamp = time.time()

        return screenshot

    async def _capture_via_hal(self) -> NDArray[np.uint8]:
        """Capture screenshot via HAL.

        Returns:
            Screenshot as numpy array.
        """
        if hasattr(self._hal, "capture_screenshot"):
            screenshot = await self._hal.capture_screenshot()
            return np.array(screenshot)
        else:
            logger.warning("HAL does not have capture_screenshot method")
            return self._capture_direct()

    def _capture_direct(self) -> NDArray[np.uint8]:
        """Capture screenshot directly using mss.

        Returns:
            Screenshot as numpy array.
        """
        try:
            import mss

            with mss.mss() as sct:
                monitor = sct.monitors[0]  # Primary monitor
                screenshot = sct.grab(monitor)

                # Convert to numpy array (BGRA)
                img = np.array(screenshot)

                # Convert BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                return img

        except ImportError:
            logger.error("mss not available for direct screenshot capture")
            # Return empty image
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def annotate(
        self,
        screenshot: NDArray[np.uint8],
        annotations: list[dict[str, Any]],
    ) -> NDArray[np.uint8]:
        """Annotate screenshot with visual markers.

        Args:
            screenshot: Screenshot to annotate.
            annotations: List of annotation dicts with keys:
                - bounds: BoundingBox or (x, y, w, h)
                - label: Optional text label
                - color: Optional BGR color tuple
                - type: 'match', 'expected', 'error'

        Returns:
            Annotated screenshot copy.
        """
        annotated = screenshot.copy()

        # Default colors
        colors = {
            "match": (0, 255, 0),  # Green
            "expected": (255, 255, 0),  # Cyan
            "error": (0, 0, 255),  # Red
            "info": (255, 255, 255),  # White
        }

        thickness = 2
        if self._config is not None:
            thickness = self._config.screenshot.annotation_thickness

        for ann in annotations:
            # Get bounds
            bounds = ann.get("bounds")
            if bounds is None:
                continue

            if isinstance(bounds, BoundingBox):
                x, y, w, h = bounds.x, bounds.y, bounds.width, bounds.height
            else:
                x, y, w, h = bounds

            # Get color
            ann_type = ann.get("type", "info")
            color = ann.get("color", colors.get(ann_type, colors["info"]))

            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

            # Draw label if present
            label = ann.get("label")
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_thickness = 1

                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, text_thickness
                )

                # Draw background
                cv2.rectangle(
                    annotated,
                    (x, y - text_h - 5),
                    (x + text_w + 4, y),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    annotated,
                    label,
                    (x + 2, y - 3),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    text_thickness,
                )

        return annotated

    def annotate_match(
        self,
        screenshot: NDArray[np.uint8],
        match: "LocatorMatch",
        label: str | None = None,
    ) -> NDArray[np.uint8]:
        """Annotate screenshot with a single match.

        Args:
            screenshot: Screenshot to annotate.
            match: Match to highlight.
            label: Optional label text.

        Returns:
            Annotated screenshot.
        """
        return self.annotate(
            screenshot,
            [
                {
                    "bounds": match.bounds,
                    "label": label or f"{match.confidence:.0%}",
                    "type": "match",
                }
            ],
        )

    def annotate_failure(
        self,
        screenshot: NDArray[np.uint8],
        expected_bounds: BoundingBox | None = None,
        best_match: "LocatorMatch | None" = None,
        message: str | None = None,
    ) -> NDArray[np.uint8]:
        """Annotate screenshot with failure information.

        Args:
            screenshot: Screenshot to annotate.
            expected_bounds: Expected element bounds.
            best_match: Best (insufficient) match found.
            message: Error message to display.

        Returns:
            Annotated screenshot.
        """
        annotations = []

        if expected_bounds is not None:
            annotations.append(
                {
                    "bounds": expected_bounds,
                    "label": "Expected",
                    "type": "expected",
                }
            )

        if best_match is not None:
            annotations.append(
                {
                    "bounds": best_match.bounds,
                    "label": f"Best match: {best_match.confidence:.0%}",
                    "type": "error",
                }
            )

        annotated = self.annotate(screenshot, annotations)

        # Add message at bottom
        if message:
            h, w = annotated.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6

            # Draw background bar
            cv2.rectangle(annotated, (0, h - 30), (w, h), (50, 50, 50), -1)

            # Draw text
            cv2.putText(
                annotated,
                message[:100],  # Truncate long messages
                (10, h - 10),
                font,
                font_scale,
                (255, 255, 255),
                1,
            )

        return annotated

    def save(
        self,
        screenshot: NDArray[np.uint8],
        name: str | None = None,
        prefix: str = "",
        suffix: str = "",
    ) -> Path:
        """Save screenshot to file.

        Args:
            screenshot: Screenshot to save.
            name: Optional filename (without extension).
            prefix: Filename prefix.
            suffix: Filename suffix.

        Returns:
            Path to saved file.
        """
        # Ensure directory exists
        self._save_directory.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"{prefix}{timestamp}{suffix}"

        filepath = self._save_directory / f"{name}.png"

        # Save image
        cv2.imwrite(str(filepath), screenshot)
        logger.debug(f"Saved screenshot: {filepath}")

        return filepath

    def save_failure(
        self,
        screenshot: NDArray[np.uint8],
        annotated: NDArray[np.uint8] | None = None,
        assertion_id: str | None = None,
    ) -> tuple[Path, Path | None]:
        """Save failure screenshot and optional annotated version.

        Args:
            screenshot: Original screenshot.
            annotated: Annotated screenshot.
            assertion_id: Assertion ID for filename.

        Returns:
            Tuple of (screenshot_path, annotated_path or None).
        """
        prefix = f"failure_{assertion_id}_" if assertion_id else "failure_"

        screenshot_path = self.save(screenshot, prefix=prefix, suffix="_original")

        annotated_path = None
        if annotated is not None:
            annotated_path = self.save(annotated, prefix=prefix, suffix="_annotated")

        return screenshot_path, annotated_path

    def cleanup_old(self, max_files: int | None = None) -> int:
        """Remove old screenshot files.

        Args:
            max_files: Maximum files to keep.

        Returns:
            Number of files removed.
        """
        if max_files is None:
            if self._config is not None:
                max_files = self._config.screenshot.max_saved
            else:
                max_files = 100

        if not self._save_directory.exists():
            return 0

        # Get all screenshots
        files = sorted(
            self._save_directory.glob("*.png"),
            key=lambda f: f.stat().st_mtime,
        )

        # Remove oldest files
        removed = 0
        while len(files) > max_files:
            oldest = files.pop(0)
            try:
                oldest.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"Failed to remove {oldest}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} old screenshots")

        return removed


# Global screenshot manager instance
_screenshot_manager: ScreenshotManager | None = None


def get_screenshot_manager(config: "VisionConfig | None" = None) -> ScreenshotManager:
    """Get the global screenshot manager.

    Args:
        config: Optional configuration.

    Returns:
        ScreenshotManager instance.
    """
    global _screenshot_manager
    if _screenshot_manager is None:
        _screenshot_manager = ScreenshotManager(config=config)
    return _screenshot_manager


__all__ = [
    "ScreenshotManager",
    "get_screenshot_manager",
]
