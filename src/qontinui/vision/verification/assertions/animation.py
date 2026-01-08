"""Animation and stability detection assertions.

Provides assertions for detecting when animations complete
and when the UI becomes stable (no changes).
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    AssertionType,
    BoundingBox,
)

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Result of stability check."""

    is_stable: bool
    frames_analyzed: int
    stability_duration_ms: int
    max_change_detected: float
    change_history: list[float] = field(default_factory=list)
    unstable_regions: list[BoundingBox] = field(default_factory=list)


@dataclass
class AnimationResult:
    """Result of animation detection."""

    animation_detected: bool
    animation_stopped: bool
    frames_analyzed: int
    duration_ms: int
    change_rate: float  # Changes per second
    final_frame: NDArray[np.uint8] | None = None


class StabilityDetector:
    """Detects UI stability by analyzing frame-to-frame changes.

    Usage:
        detector = StabilityDetector(config)

        # Wait for stability
        result = await detector.wait_for_stability(
            screenshot_callback,
            stability_threshold=0.001,
            stability_duration_ms=500,
        )
    """

    def __init__(self, config: "VisionConfig | None" = None) -> None:
        """Initialize stability detector.

        Args:
            config: Vision configuration.
        """
        self._config = config

    def _get_default_threshold(self) -> float:
        """Get default stability threshold.

        Returns:
            Threshold (0.0-1.0) for change detection.
        """
        if self._config is not None:
            return self._config.wait.stability_threshold
        return 0.001  # 0.1% change allowed

    def _get_default_duration(self) -> int:
        """Get default stability duration.

        Returns:
            Duration in ms that must be stable.
        """
        if self._config is not None:
            return self._config.wait.stability_duration_ms
        return 500

    def _get_poll_interval(self) -> int:
        """Get poll interval.

        Returns:
            Interval in ms between captures.
        """
        if self._config is not None:
            return self._config.wait.poll_interval_ms
        return 100

    def calculate_change(
        self,
        frame1: NDArray[np.uint8],
        frame2: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> float:
        """Calculate change between two frames.

        Args:
            frame1: First frame.
            frame2: Second frame.
            region: Optional region to analyze.

        Returns:
            Change ratio (0.0-1.0).
        """
        # Crop to region if specified
        if region is not None:
            frame1 = frame1[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ]
            frame2 = frame2[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ]

        # Ensure same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Convert to grayscale for comparison
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Count changed pixels (above noise threshold)
        _, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size

        return changed_pixels / total_pixels

    def find_changed_regions(
        self,
        frame1: NDArray[np.uint8],
        frame2: NDArray[np.uint8],
    ) -> list[BoundingBox]:
        """Find regions that changed between frames.

        Args:
            frame1: First frame.
            frame2: Second frame.

        Returns:
            List of changed regions.
        """
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2

        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:  # Filter noise
                regions.append(BoundingBox(x=x, y=y, width=w, height=h))

        return regions

    async def wait_for_stability(
        self,
        screenshot_callback: Callable[[], Any],
        stability_threshold: float | None = None,
        stability_duration_ms: int | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> StabilityResult:
        """Wait for UI to become stable.

        Args:
            screenshot_callback: Async function to capture screenshots.
            stability_threshold: Max change ratio to consider stable.
            stability_duration_ms: How long must be stable.
            timeout_ms: Maximum wait time.
            region: Optional region to monitor.

        Returns:
            Stability result.
        """
        threshold = stability_threshold or self._get_default_threshold()
        duration = stability_duration_ms or self._get_default_duration()
        timeout = timeout_ms or 10000
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        stable_since: float | None = None
        last_frame: NDArray[np.uint8] | None = None
        frames_analyzed = 0
        change_history: list[float] = []
        max_change = 0.0

        while True:
            # Capture frame
            current_frame = await screenshot_callback()
            frames_analyzed += 1

            if last_frame is not None:
                # Calculate change
                change = self.calculate_change(last_frame, current_frame, region)
                change_history.append(change)
                max_change = max(max_change, change)

                if change <= threshold:
                    # Frame is stable
                    if stable_since is None:
                        stable_since = time.monotonic()

                    # Check if stable long enough
                    stable_duration = (time.monotonic() - stable_since) * 1000
                    if stable_duration >= duration:
                        return StabilityResult(
                            is_stable=True,
                            frames_analyzed=frames_analyzed,
                            stability_duration_ms=int(stable_duration),
                            max_change_detected=max_change,
                            change_history=change_history,
                        )
                else:
                    # Not stable, reset
                    stable_since = None

            last_frame = current_frame

            # Check timeout
            elapsed = (time.monotonic() - start_time) * 1000
            if elapsed >= timeout:
                # Find unstable regions from last comparison
                unstable_regions = []
                if last_frame is not None and current_frame is not None:
                    unstable_regions = self.find_changed_regions(last_frame, current_frame)

                return StabilityResult(
                    is_stable=False,
                    frames_analyzed=frames_analyzed,
                    stability_duration_ms=0,
                    max_change_detected=max_change,
                    change_history=change_history,
                    unstable_regions=unstable_regions,
                )

            await asyncio.sleep(poll_interval / 1000)


class AnimationDetector:
    """Detects and waits for animations to complete.

    Usage:
        detector = AnimationDetector(config)

        # Wait for animation to stop
        result = await detector.wait_for_animation_end(
            screenshot_callback,
            timeout_ms=5000,
        )
    """

    def __init__(self, config: "VisionConfig | None" = None) -> None:
        """Initialize animation detector.

        Args:
            config: Vision configuration.
        """
        self._config = config
        self._stability_detector = StabilityDetector(config)

    def _get_poll_interval(self) -> int:
        """Get poll interval.

        Returns:
            Interval in ms.
        """
        if self._config is not None:
            return self._config.wait.poll_interval_ms
        return 50  # Faster for animation detection

    async def detect_animation(
        self,
        screenshot_callback: Callable[[], Any],
        sample_count: int = 5,
        sample_interval_ms: int = 50,
        region: BoundingBox | None = None,
    ) -> AnimationResult:
        """Detect if animation is occurring.

        Args:
            screenshot_callback: Async function to capture screenshots.
            sample_count: Number of frames to sample.
            sample_interval_ms: Interval between samples.
            region: Optional region to monitor.

        Returns:
            Animation detection result.
        """
        start_time = time.monotonic()
        frames: list[NDArray[np.uint8]] = []
        changes: list[float] = []

        # Collect frames
        for _ in range(sample_count):
            frame = await screenshot_callback()
            frames.append(frame)
            await asyncio.sleep(sample_interval_ms / 1000)

        # Calculate changes between consecutive frames
        for i in range(1, len(frames)):
            change = self._stability_detector.calculate_change(frames[i - 1], frames[i], region)
            changes.append(change)

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Animation detected if average change is above threshold
        avg_change = sum(changes) / len(changes) if changes else 0
        animation_threshold = 0.005  # 0.5% change indicates animation

        animation_detected = avg_change > animation_threshold

        # Calculate change rate (changes per second)
        total_time_s = (sample_count * sample_interval_ms) / 1000
        change_rate = sum(1 for c in changes if c > animation_threshold) / total_time_s

        return AnimationResult(
            animation_detected=animation_detected,
            animation_stopped=not animation_detected,
            frames_analyzed=sample_count,
            duration_ms=duration_ms,
            change_rate=change_rate,
            final_frame=frames[-1] if frames else None,
        )

    async def wait_for_animation_end(
        self,
        screenshot_callback: Callable[[], Any],
        timeout_ms: int | None = None,
        stability_duration_ms: int = 200,
        region: BoundingBox | None = None,
    ) -> AnimationResult:
        """Wait for animation to complete.

        Args:
            screenshot_callback: Async function to capture screenshots.
            timeout_ms: Maximum wait time.
            stability_duration_ms: How long must be stable to consider done.
            region: Optional region to monitor.

        Returns:
            Animation result.
        """
        timeout = timeout_ms or 10000
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        frames_analyzed = 0
        last_frame: NDArray[np.uint8] | None = None
        stable_since: float | None = None
        total_changes = 0

        while True:
            current_frame = await screenshot_callback()
            frames_analyzed += 1

            if last_frame is not None:
                change = self._stability_detector.calculate_change(
                    last_frame, current_frame, region
                )

                if change > 0.001:
                    total_changes += 1
                    stable_since = None
                else:
                    if stable_since is None:
                        stable_since = time.monotonic()

                    stable_duration = (time.monotonic() - stable_since) * 1000
                    if stable_duration >= stability_duration_ms:
                        duration_ms = int((time.monotonic() - start_time) * 1000)
                        change_rate = total_changes / (duration_ms / 1000)

                        return AnimationResult(
                            animation_detected=total_changes > 0,
                            animation_stopped=True,
                            frames_analyzed=frames_analyzed,
                            duration_ms=duration_ms,
                            change_rate=change_rate,
                            final_frame=current_frame,
                        )

            last_frame = current_frame

            # Check timeout
            elapsed = (time.monotonic() - start_time) * 1000
            if elapsed >= timeout:
                change_rate = total_changes / (elapsed / 1000)

                return AnimationResult(
                    animation_detected=total_changes > 0,
                    animation_stopped=False,
                    frames_analyzed=frames_analyzed,
                    duration_ms=int(elapsed),
                    change_rate=change_rate,
                    final_frame=current_frame,
                )

            await asyncio.sleep(poll_interval / 1000)


class AnimationAssertion:
    """Assertions for animation and stability.

    Usage:
        assertion = AnimationAssertion(config)

        # Assert animation has stopped
        result = await assertion.to_stop_animating(screenshot_callback)

        # Assert UI is stable
        result = await assertion.to_be_stable(screenshot_callback)
    """

    def __init__(self, config: "VisionConfig | None" = None) -> None:
        """Initialize animation assertion.

        Args:
            config: Vision configuration.
        """
        self._config = config
        self._stability_detector = StabilityDetector(config)
        self._animation_detector = AnimationDetector(config)

    async def to_stop_animating(
        self,
        screenshot_callback: Callable[[], Any],
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert animation has stopped.

        Args:
            screenshot_callback: Async function to capture screenshots.
            timeout_ms: Maximum wait time.
            region: Optional region to monitor.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        result = await self._animation_detector.wait_for_animation_end(
            screenshot_callback=screenshot_callback,
            timeout_ms=timeout_ms,
            region=region,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if result.animation_stopped:
            return AssertionResult(
                assertion_id="animation_stopped",
                locator_value="screen" if region is None else f"region({region.x},{region.y})",
                assertion_type=AssertionType.ANIMATION,
                assertion_method="to_stop_animating",
                status=AssertionStatus.PASSED,
                message=f"Animation stopped after {result.duration_ms}ms",
                expected_value="no animation",
                actual_value="stable",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="animation_stopped",
                locator_value="screen" if region is None else f"region({region.x},{region.y})",
                assertion_type=AssertionType.ANIMATION,
                assertion_method="to_stop_animating",
                status=AssertionStatus.FAILED,
                message=f"Animation still running ({result.change_rate:.1f} changes/sec)",
                expected_value="no animation",
                actual_value=f"{result.change_rate:.1f} changes/sec",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_stable(
        self,
        screenshot_callback: Callable[[], Any],
        stability_threshold: float | None = None,
        stability_duration_ms: int | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert UI is stable (no changes).

        Args:
            screenshot_callback: Async function to capture screenshots.
            stability_threshold: Max change ratio.
            stability_duration_ms: Required stable duration.
            timeout_ms: Maximum wait time.
            region: Optional region to monitor.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        result = await self._stability_detector.wait_for_stability(
            screenshot_callback=screenshot_callback,
            stability_threshold=stability_threshold,
            stability_duration_ms=stability_duration_ms,
            timeout_ms=timeout_ms,
            region=region,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if result.is_stable:
            return AssertionResult(
                assertion_id="ui_stable",
                locator_value="screen" if region is None else f"region({region.x},{region.y})",
                assertion_type=AssertionType.ANIMATION,
                assertion_method="to_be_stable",
                status=AssertionStatus.PASSED,
                message=f"UI stable for {result.stability_duration_ms}ms",
                expected_value="stable",
                actual_value="stable",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="ui_stable",
                locator_value="screen" if region is None else f"region({region.x},{region.y})",
                assertion_type=AssertionType.ANIMATION,
                assertion_method="to_be_stable",
                status=AssertionStatus.FAILED,
                message=f"UI not stable (max change: {result.max_change_detected:.2%})",
                expected_value="stable",
                actual_value=f"{len(result.unstable_regions)} unstable regions",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_animating(
        self,
        screenshot_callback: Callable[[], Any],
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert animation is currently occurring.

        Args:
            screenshot_callback: Async function to capture screenshots.
            region: Optional region to monitor.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        result = await self._animation_detector.detect_animation(
            screenshot_callback=screenshot_callback,
            region=region,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if result.animation_detected:
            return AssertionResult(
                assertion_id="animation_active",
                locator_value="screen" if region is None else f"region({region.x},{region.y})",
                assertion_type=AssertionType.ANIMATION,
                assertion_method="to_be_animating",
                status=AssertionStatus.PASSED,
                message=f"Animation detected ({result.change_rate:.1f} changes/sec)",
                expected_value="animating",
                actual_value=f"{result.change_rate:.1f} changes/sec",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="animation_active",
                locator_value="screen" if region is None else f"region({region.x},{region.y})",
                assertion_type=AssertionType.ANIMATION,
                assertion_method="to_be_animating",
                status=AssertionStatus.FAILED,
                message="No animation detected",
                expected_value="animating",
                actual_value="static",
                matches_found=0,
                duration_ms=elapsed_ms,
            )


__all__ = [
    "AnimationAssertion",
    "AnimationDetector",
    "AnimationResult",
    "StabilityDetector",
    "StabilityResult",
]
