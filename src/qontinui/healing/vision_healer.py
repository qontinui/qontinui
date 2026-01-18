"""Vision-based self-healing for failed element lookups.

Coordinates healing attempts using vision LLM when template matching fails.
Integrates with action cache to store healed locations.

This module emits healing events for monitoring and metrics in qontinui-runner.
"""

import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..reporting.events import EventType, emit_event
from ..reporting.schemas import (
    HealingAttemptData,
    HealingMetricsData,
    VisualValidationData,
)
from .healing_config import HealingConfig
from .healing_types import (
    ElementLocation,
    HealingContext,
    HealingResult,
    HealingStrategy,
    LLMMode,
)

if TYPE_CHECKING:
    from ..cache import ActionCache

logger = logging.getLogger(__name__)


class VisionHealer:
    """Coordinates vision-based self-healing for failed element lookups.

    When template matching fails, VisionHealer can:
    1. Try visual search in nearby areas (no LLM)
    2. Use vision LLM to locate the element (if enabled)
    3. Cache healed locations for future use

    Default behavior is NO LLM (llm_mode=DISABLED). User must explicitly
    enable LLM healing by configuring HealingConfig.

    Attributes:
        config: Healing configuration.
        cache: Optional action cache for storing healed locations.
    """

    def __init__(
        self,
        config: HealingConfig | None = None,
        cache: "ActionCache | None" = None,
    ) -> None:
        """Initialize vision healer.

        Args:
            config: Healing configuration. Defaults to disabled.
            cache: Optional action cache for persisting healed locations.
        """
        self.config = config or HealingConfig.disabled()
        self.cache = cache

        # Statistics
        self._total_attempts = 0
        self._successful_heals = 0
        self._llm_calls = 0

    def heal(
        self,
        screenshot: np.ndarray,
        context: HealingContext,
        pattern: "np.ndarray | None" = None,
    ) -> HealingResult:
        """Attempt to heal a failed element lookup.

        Tries multiple strategies in order:
        1. Visual search (nearby area scan)
        2. Vision LLM (if enabled)

        Args:
            screenshot: Current screen capture (BGR numpy array).
            context: Context about the failed lookup.
            pattern: Optional original pattern for visual search.

        Returns:
            HealingResult with success/failure and location if found.
        """
        start_time = time.time()
        self._total_attempts += 1

        attempts: list[tuple[HealingStrategy, str]] = []

        # Emit healing started event
        self._emit_healing_started(context)

        # Strategy 1: Visual search (no LLM, always available)
        if pattern is not None:
            self._emit_strategy_attempted("visual_search", context)
            location = self._try_visual_search(screenshot, pattern, context)
            if location:
                duration_ms = (time.time() - start_time) * 1000
                self._successful_heals += 1
                result = HealingResult(
                    success=True,
                    strategy=HealingStrategy.VISUAL_SEARCH,
                    location=location,
                    message="Found element via expanded visual search",
                    attempts=attempts,
                    duration_ms=duration_ms,
                )
                self._emit_healing_succeeded(result, context, "visual_search")
                return result
            attempts.append((HealingStrategy.VISUAL_SEARCH, "Pattern not found in expanded search"))
            self._emit_strategy_failed("visual_search", "Pattern not found in expanded search")

        # Strategy 2: Vision LLM (only if enabled)
        if self.config.llm_mode != LLMMode.DISABLED:
            self._emit_strategy_attempted("llm_vision", context)
            location, llm_message = self._try_llm_healing(screenshot, context)
            if location:
                duration_ms = (time.time() - start_time) * 1000
                self._successful_heals += 1
                self._llm_calls += 1
                result = HealingResult(
                    success=True,
                    strategy=HealingStrategy.LLM_VISION,
                    location=location,
                    message="Found element via vision LLM",
                    attempts=attempts,
                    llm_tokens_used=100,  # Approximate
                    duration_ms=duration_ms,
                )
                self._emit_healing_succeeded(result, context, "llm_vision")
                return result
            attempts.append((HealingStrategy.LLM_VISION, llm_message))
            self._emit_strategy_failed("llm_vision", llm_message)

        # All strategies failed
        duration_ms = (time.time() - start_time) * 1000
        result = HealingResult(
            success=False,
            strategy=HealingStrategy.FAILED,
            message="All healing strategies failed",
            attempts=attempts,
            duration_ms=duration_ms,
        )
        self._emit_healing_failed(result, context, attempts)
        return result

    def _try_visual_search(
        self,
        screenshot: np.ndarray,
        pattern: np.ndarray,
        context: HealingContext,
    ) -> ElementLocation | None:
        """Try to find pattern with expanded search parameters.

        Searches with:
        - Lower similarity threshold
        - Multiple scales
        - Expanded search regions

        Args:
            screenshot: Screen capture.
            pattern: Pattern to search for.
            context: Healing context.

        Returns:
            ElementLocation if found, None otherwise.
        """
        # Prepare pattern (remove alpha if present)
        if len(pattern.shape) == 3 and pattern.shape[2] == 4:
            pattern = pattern[:, :, :3]

        # Try at multiple similarity thresholds
        for threshold in [0.7, 0.6, 0.5]:
            result = cv2.matchTemplate(screenshot, pattern, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                # Found with lower threshold
                x = max_loc[0] + pattern.shape[1] // 2
                y = max_loc[1] + pattern.shape[0] // 2

                logger.info(
                    f"Visual search found element at ({x}, {y}) with "
                    f"similarity {max_val:.3f} (threshold {threshold})"
                )

                return ElementLocation(
                    x=x,
                    y=y,
                    confidence=max_val,
                    region=(max_loc[0], max_loc[1], pattern.shape[1], pattern.shape[0]),
                    description=f"Found at lower threshold {threshold}",
                )

        # Try scaled versions
        for scale in [0.9, 1.1, 0.8, 1.2]:
            scaled_w = int(pattern.shape[1] * scale)
            scaled_h = int(pattern.shape[0] * scale)

            if scaled_w < 10 or scaled_h < 10:
                continue

            scaled_pattern = cv2.resize(pattern, (scaled_w, scaled_h))

            result = cv2.matchTemplate(screenshot, scaled_pattern, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= 0.7:
                x = max_loc[0] + scaled_w // 2
                y = max_loc[1] + scaled_h // 2

                logger.info(
                    f"Visual search found scaled element at ({x}, {y}) with "
                    f"similarity {max_val:.3f} at scale {scale}"
                )

                return ElementLocation(
                    x=x,
                    y=y,
                    confidence=max_val,
                    region=(max_loc[0], max_loc[1], scaled_w, scaled_h),
                    description=f"Found at scale {scale}",
                )

        return None

    def _try_llm_healing(
        self,
        screenshot: np.ndarray,
        context: HealingContext,
    ) -> tuple[ElementLocation | None, str]:
        """Try to find element using vision LLM.

        Args:
            screenshot: Screen capture.
            context: Healing context.

        Returns:
            Tuple of (location or None, message).
        """
        client = self.config.get_client()

        if not client.is_available:
            return None, "LLM client not available"

        # Convert screenshot to PNG bytes
        try:
            success, png_data = cv2.imencode(".png", screenshot)
            if not success:
                return None, "Failed to encode screenshot"
            screenshot_bytes = png_data.tobytes()
        except Exception as e:
            return None, f"Screenshot encoding error: {e}"

        # Update context with screenshot info
        context.screenshot_shape = (screenshot.shape[0], screenshot.shape[1])

        # Call LLM
        try:
            location = client.find_element(screenshot_bytes, context)

            if location:
                # Validate coordinates are within bounds
                if 0 <= location.x < screenshot.shape[1] and 0 <= location.y < screenshot.shape[0]:
                    logger.info(
                        f"LLM found element at ({location.x}, {location.y}) "
                        f"with confidence {location.confidence:.2f}"
                    )
                    return location, "Found by LLM"
                else:
                    return (
                        None,
                        f"LLM returned out-of-bounds coordinates: ({location.x}, {location.y})",
                    )

            return None, "LLM could not locate element"

        except Exception as e:
            logger.error(f"LLM healing error: {e}")
            return None, f"LLM error: {e}"

    def get_stats(self) -> dict[str, int | float]:
        """Get healing statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_attempts": self._total_attempts,
            "successful_heals": self._successful_heals,
            "llm_calls": self._llm_calls,
            "success_rate": (
                self._successful_heals / self._total_attempts * 100
                if self._total_attempts > 0
                else 0.0
            ),
        }

    # =========================================================================
    # Event Emission Methods
    # =========================================================================

    def _emit_healing_started(self, context: HealingContext) -> None:
        """Emit healing started event.

        Args:
            context: Healing context
        """
        try:
            emit_event(
                EventType.HEALING_STARTED,
                data={
                    "pattern_id": context.original_description,
                    "action_type": context.action_type,
                    "failure_reason": context.failure_reason,
                    "state_id": context.state_id,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit healing started event: {e}")

    def _emit_strategy_attempted(self, strategy: str, context: HealingContext) -> None:
        """Emit strategy attempted event.

        Args:
            strategy: Strategy name being attempted
            context: Healing context
        """
        try:
            emit_event(
                EventType.HEALING_STRATEGY_ATTEMPTED,
                data={
                    "pattern_id": context.original_description,
                    "strategy": strategy,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit strategy attempted event: {e}")

    def _emit_strategy_failed(self, strategy: str, reason: str) -> None:
        """Emit strategy failed event.

        Args:
            strategy: Strategy name that failed
            reason: Failure reason
        """
        try:
            emit_event(
                EventType.HEALING_STRATEGY_FAILED,
                data={
                    "strategy": strategy,
                    "error_message": reason,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit strategy failed event: {e}")

    def _emit_healing_succeeded(
        self,
        result: HealingResult,
        context: HealingContext,
        strategy: str,
    ) -> None:
        """Emit healing succeeded event.

        Args:
            result: Healing result
            context: Healing context
            strategy: Strategy that succeeded
        """
        try:
            event_data = HealingAttemptData(
                pattern_id=context.original_description,
                strategy=strategy,
                success=True,
                confidence=result.location.confidence if result.location else 0.0,
                duration_ms=result.duration_ms,
                location_x=result.location.x if result.location else None,
                location_y=result.location.y if result.location else None,
                timestamp=time.time(),
            )
            emit_event(EventType.HEALING_SUCCEEDED, data=event_data.to_dict())

            # Emit metrics update periodically
            if self._total_attempts % 5 == 0:
                self._emit_metrics_update()
        except Exception as e:
            logger.debug(f"Failed to emit healing succeeded event: {e}")

    def _emit_healing_failed(
        self,
        result: HealingResult,
        context: HealingContext,
        attempts: list[tuple[HealingStrategy, str]],
    ) -> None:
        """Emit healing failed event.

        Args:
            result: Healing result
            context: Healing context
            attempts: List of attempted strategies and their failure reasons
        """
        try:
            strategies_tried = ", ".join(str(s.value) for s, _ in attempts)
            event_data = HealingAttemptData(
                pattern_id=context.original_description,
                strategy="all_failed",
                success=False,
                duration_ms=result.duration_ms,
                error_message=f"All strategies failed: {strategies_tried}",
                timestamp=time.time(),
            )
            emit_event(EventType.HEALING_FAILED, data=event_data.to_dict())

            # Emit metrics update periodically
            if self._total_attempts % 5 == 0:
                self._emit_metrics_update()
        except Exception as e:
            logger.debug(f"Failed to emit healing failed event: {e}")

    def _emit_metrics_update(self) -> None:
        """Emit aggregate metrics update event."""
        try:
            stats = self.get_stats()
            metrics_data = HealingMetricsData(
                total_attempts=int(stats["total_attempts"]),
                successful_heals=int(stats["successful_heals"]),
                failed_heals=int(stats["total_attempts"]) - int(stats["successful_heals"]),
                healing_rate=float(stats["success_rate"]) / 100.0,
                llm_calls=int(stats["llm_calls"]),
                timestamp=time.time(),
            )
            emit_event(EventType.HEALING_METRICS_UPDATED, data=metrics_data.to_dict())
        except Exception as e:
            logger.debug(f"Failed to emit metrics update event: {e}")

    def _emit_visual_validation(
        self,
        validation_type: str,
        passed: bool,
        confidence: float,
        threshold: float,
        expected_state: str | None = None,
        actual_state: str | None = None,
    ) -> None:
        """Emit visual validation event.

        Args:
            validation_type: Type of validation performed
            passed: Whether validation passed
            confidence: Validation confidence
            threshold: Confidence threshold used
            expected_state: Expected state identifier
            actual_state: Actually detected state
        """
        try:
            validation_data = VisualValidationData(
                validation_type=validation_type,
                passed=passed,
                confidence=confidence,
                threshold=threshold,
                expected_state=expected_state,
                actual_state=actual_state,
                timestamp=time.time(),
            )
            event_type = (
                EventType.HEALING_VISUAL_VALIDATION_PASSED
                if passed
                else EventType.HEALING_VISUAL_VALIDATION_FAILED
            )
            emit_event(event_type, data=validation_data.to_dict())
        except Exception as e:
            logger.debug(f"Failed to emit visual validation event: {e}")


# Global healer instance
_default_healer: VisionHealer | None = None


def get_vision_healer() -> VisionHealer:
    """Get the global vision healer instance.

    Returns:
        VisionHealer instance (creates default if needed).
    """
    global _default_healer
    if _default_healer is None:
        _default_healer = VisionHealer()
    return _default_healer


def set_vision_healer(healer: VisionHealer) -> None:
    """Set the global vision healer instance.

    Args:
        healer: VisionHealer to use globally.
    """
    global _default_healer
    _default_healer = healer


def configure_healing(config: HealingConfig) -> VisionHealer:
    """Configure global healing with the given config.

    Convenience function to create and set a healer with custom config.

    Args:
        config: Healing configuration.

    Returns:
        The configured VisionHealer.
    """
    healer = VisionHealer(config=config)
    set_vision_healer(healer)
    return healer
