"""Self-healing manager for locators.

Manages automatic healing when primary locators fail, learns which strategies work,
and optionally updates stored patterns after successful healing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..model.element import Pattern
from ..reporting.events import EventType, emit_event
from .multi_strategy import MultiStrategyLocator, MultiStrategyResult
from .strategies import MatchResult, ScreenContext

logger = logging.getLogger(__name__)


class HealingEventType(str, Enum):
    """Types of healing events."""

    HEALING_ATTEMPTED = "healing_attempted"
    HEALING_SUCCEEDED = "healing_succeeded"
    HEALING_FAILED = "healing_failed"
    PATTERN_UPDATED = "pattern_updated"
    STRATEGY_LEARNED = "strategy_learned"


@dataclass
class HealingConfig:
    """Configuration for self-healing behavior.

    Attributes:
        auto_heal: Enable automatic healing when primary locator fails
        confidence_threshold: Minimum confidence to accept healed match
        update_on_heal: Automatically update pattern after successful healing
        max_healing_attempts: Maximum number of alternative strategies to try
        learn_successful_strategies: Track and prefer successful strategies
        emit_events: Emit events for healing operations
        fallback_strategies: List of strategy names to use as fallbacks
    """

    auto_heal: bool = True
    confidence_threshold: float = 0.7
    update_on_heal: bool = False
    max_healing_attempts: int = 5
    learn_successful_strategies: bool = True
    emit_events: bool = True
    fallback_strategies: list[str] = field(
        default_factory=lambda: ["visual", "text", "relative", "color", "structural"]
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.max_healing_attempts < 1:
            raise ValueError("max_healing_attempts must be at least 1")


@dataclass
class HealingAttempt:
    """Record of a healing attempt.

    Attributes:
        pattern_id: ID of pattern being healed
        timestamp: When healing was attempted
        primary_failed: Whether primary strategy failed
        healing_succeeded: Whether healing found a match
        successful_strategy: Name of strategy that succeeded
        confidence: Confidence of healed match
        duration: Time taken for healing
        updated_pattern: Whether pattern was updated
    """

    pattern_id: str
    timestamp: float
    primary_failed: bool
    healing_succeeded: bool
    successful_strategy: str | None = None
    confidence: float = 0.0
    duration: float = 0.0
    updated_pattern: bool = False


class HealingManager:
    """Manages self-healing for locators.

    Tracks when primary locators fail and automatically tries alternative strategies.
    Learns which strategies work for different patterns and optionally updates
    stored patterns after successful healing.

    Example:
        >>> config = HealingConfig(auto_heal=True, update_on_heal=True)
        >>> manager = HealingManager(config)
        >>>
        >>> # Try to find with healing
        >>> result = manager.find_with_healing(pattern, context)
        >>> if result.found:
        ...     if result.metadata.get('healed'):
        ...         print(f"Healed using {result.successful_strategy}")
        ...     else:
        ...         print("Found with primary strategy")
    """

    def __init__(self, config: HealingConfig | None = None) -> None:
        """Initialize healing manager.

        Args:
            config: Healing configuration (default: HealingConfig())
        """
        self.config = config or HealingConfig()

        # Create multi-strategy locator with configured strategies
        self.locator = MultiStrategyLocator.create_with_strategies(*self.config.fallback_strategies)

        # Healing history
        self._healing_history: list[HealingAttempt] = []

        # Strategy success tracking (pattern_id -> strategy_name -> success_count)
        self._strategy_success: dict[str, dict[str, int]] = {}

        # Pattern update callbacks (pattern_id -> callback)
        self._update_callbacks: dict[str, Any] = {}

    def find_with_healing(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float | None = None,
    ) -> MultiStrategyResult:
        """Find element with automatic healing on failure.

        First tries primary strategy (visual pattern matching). If that fails
        and auto_heal is enabled, tries alternative strategies.

        Args:
            target: Pattern or configuration dict
            context: Screen context
            min_confidence: Override confidence threshold (uses config if None)

        Returns:
            MultiStrategyResult with healing metadata
        """
        start_time = time.time()
        confidence = min_confidence or self.config.confidence_threshold

        # Extract pattern ID for tracking
        pattern_id = self._get_pattern_id(target)

        # Emit healing attempt event
        if self.config.emit_events:
            self._emit_event(
                HealingEventType.HEALING_ATTEMPTED,
                {
                    "pattern_id": pattern_id,
                    "auto_heal": self.config.auto_heal,
                    "confidence_threshold": confidence,
                },
            )

        # Try multi-strategy search
        result = self.locator.find(target, context, confidence, stop_on_first=True)

        # Record healing attempt
        healing_attempt = HealingAttempt(
            pattern_id=pattern_id,
            timestamp=start_time,
            primary_failed=(
                result.successful_strategy != "VisualPattern" if result.found else True
            ),
            healing_succeeded=result.found,
            successful_strategy=result.successful_strategy,
            confidence=result.confidence,
            duration=time.time() - start_time,
        )

        # Check if healing was used (non-primary strategy succeeded)
        healed = result.found and result.successful_strategy != "VisualPattern"

        if healed:
            logger.info(
                f"Healing succeeded for pattern {pattern_id} "
                f"using {result.successful_strategy} (confidence={result.confidence:.3f})"
            )

            # Learn successful strategy
            if self.config.learn_successful_strategies:
                self._record_successful_strategy(pattern_id, result.successful_strategy)  # type: ignore[arg-type]

            # Update pattern if configured
            if self.config.update_on_heal and isinstance(target, Pattern):
                updated = self._update_pattern(target, result.match_result, context)
                healing_attempt.updated_pattern = updated

            # Emit success event
            if self.config.emit_events:
                self._emit_event(
                    HealingEventType.HEALING_SUCCEEDED,
                    {
                        "pattern_id": pattern_id,
                        "strategy": result.successful_strategy,
                        "confidence": result.confidence,
                        "updated_pattern": healing_attempt.updated_pattern,
                    },
                )

        elif not result.found:
            logger.warning(f"Healing failed for pattern {pattern_id} - no strategy succeeded")

            # Emit failure event
            if self.config.emit_events:
                self._emit_event(
                    HealingEventType.HEALING_FAILED,
                    {
                        "pattern_id": pattern_id,
                        "attempts": len(result.attempts),
                    },
                )

        # Record in history
        self._healing_history.append(healing_attempt)

        # Add healing metadata to result
        result.metadata["healed"] = healed
        result.metadata["primary_failed"] = healing_attempt.primary_failed

        return result

    def register_update_callback(self, pattern_id: str, callback: Any) -> None:
        """Register callback for pattern updates.

        Args:
            pattern_id: Pattern ID to monitor
            callback: Callback function (pattern, new_pixel_data) -> None
        """
        self._update_callbacks[pattern_id] = callback

    def get_healing_history(self) -> list[HealingAttempt]:
        """Get history of healing attempts.

        Returns:
            List of all healing attempts
        """
        return self._healing_history.copy()

    def get_healing_stats(self) -> dict[str, Any]:
        """Get statistics about healing operations.

        Returns:
            Dict with healing statistics
        """
        if not self._healing_history:
            return {
                "total_attempts": 0,
                "healing_successes": 0,
                "healing_failures": 0,
                "healing_rate": 0.0,
                "patterns_updated": 0,
            }

        total = len(self._healing_history)
        successes = sum(1 for a in self._healing_history if a.healing_succeeded)
        failures = total - successes
        patterns_updated = sum(1 for a in self._healing_history if a.updated_pattern)

        return {
            "total_attempts": total,
            "healing_successes": successes,
            "healing_failures": failures,
            "healing_rate": float(successes) / float(total) if total > 0 else 0.0,
            "patterns_updated": patterns_updated,
        }

    def get_strategy_preferences(self, pattern_id: str) -> dict[str, int]:
        """Get learned strategy preferences for a pattern.

        Args:
            pattern_id: Pattern ID

        Returns:
            Dict mapping strategy names to success counts
        """
        return self._strategy_success.get(pattern_id, {}).copy()

    def clear_history(self) -> None:
        """Clear healing history and learned preferences."""
        self._healing_history.clear()
        self._strategy_success.clear()

    def _get_pattern_id(self, target: Pattern | dict[str, Any]) -> str:
        """Extract pattern ID from target.

        Args:
            target: Pattern or config dict

        Returns:
            Pattern ID string
        """
        if isinstance(target, Pattern):
            return target.id
        elif isinstance(target, dict):
            return str(target.get("id", "unknown"))
        return "unknown"

    def _record_successful_strategy(self, pattern_id: str, strategy_name: str) -> None:
        """Record successful strategy for learning.

        Args:
            pattern_id: Pattern ID
            strategy_name: Name of successful strategy
        """
        if pattern_id not in self._strategy_success:
            self._strategy_success[pattern_id] = {}

        if strategy_name not in self._strategy_success[pattern_id]:
            self._strategy_success[pattern_id][strategy_name] = 0

        self._strategy_success[pattern_id][strategy_name] += 1

        logger.debug(
            f"Recorded successful strategy {strategy_name} for pattern {pattern_id} "
            f"(count={self._strategy_success[pattern_id][strategy_name]})"
        )

        # Emit learning event
        if self.config.emit_events:
            self._emit_event(
                HealingEventType.STRATEGY_LEARNED,
                {
                    "pattern_id": pattern_id,
                    "strategy": strategy_name,
                    "success_count": self._strategy_success[pattern_id][strategy_name],
                },
            )

    def _update_pattern(
        self,
        pattern: Pattern,
        match_result: MatchResult | None,
        context: ScreenContext,
    ) -> bool:
        """Update pattern with new pixel data from healed match.

        Args:
            pattern: Pattern to update
            match_result: Match result with region
            context: Screen context with screenshot

        Returns:
            True if pattern was updated
        """
        if not match_result or not match_result.region:
            return False

        try:
            # Extract new pixel data from screenshot
            region = match_result.region
            new_pixel_data = context.screenshot[
                region.y : region.y + region.height, region.x : region.x + region.width
            ]

            # Update pattern pixel data
            old_pixel_data = pattern.pixel_data.copy()
            pattern.pixel_data = new_pixel_data

            # Keep same mask (or regenerate if needed)
            # For now, keep existing mask shape
            if pattern.mask.shape[:2] != new_pixel_data.shape[:2]:
                import numpy as np

                pattern.mask = np.ones(new_pixel_data.shape[:2], dtype=np.float32)

            logger.info(f"Updated pattern {pattern.id} with new pixel data from healed match")

            # Call update callback if registered
            if pattern.id in self._update_callbacks:
                try:
                    self._update_callbacks[pattern.id](pattern, new_pixel_data)
                except Exception as e:
                    logger.error(f"Pattern update callback error: {e}", exc_info=True)

            # Emit update event
            if self.config.emit_events:
                self._emit_event(
                    HealingEventType.PATTERN_UPDATED,
                    {
                        "pattern_id": pattern.id,
                        "old_size": old_pixel_data.shape,
                        "new_size": new_pixel_data.shape,
                        "strategy": match_result.strategy_name,
                    },
                )

            return True

        except Exception as e:
            logger.error(f"Failed to update pattern {pattern.id}: {e}", exc_info=True)
            return False

    def _emit_event(self, event_type: HealingEventType, data: dict[str, Any]) -> None:
        """Emit healing event.

        Args:
            event_type: Type of healing event
            data: Event data
        """
        try:
            # Map healing event to generic event type
            if event_type == HealingEventType.HEALING_SUCCEEDED:
                generic_type = EventType.ACTION_COMPLETED
            elif event_type == HealingEventType.HEALING_FAILED:
                generic_type = EventType.ACTION_FAILED
            else:
                generic_type = EventType.ACTION_STARTED

            emit_event(
                generic_type,
                data={
                    "healing_event": event_type.value,
                    **data,
                },
            )
        except Exception as e:
            logger.error(f"Failed to emit healing event: {e}", exc_info=True)

    @classmethod
    def create_with_defaults(cls) -> HealingManager:
        """Create healing manager with default configuration.

        Returns:
            HealingManager with default settings
        """
        return cls(HealingConfig())

    @classmethod
    def create_aggressive(cls) -> HealingManager:
        """Create healing manager with aggressive healing settings.

        Aggressive mode:
        - Auto-heal enabled
        - Updates patterns on heal
        - Lower confidence threshold
        - More fallback strategies

        Returns:
            HealingManager with aggressive settings
        """
        config = HealingConfig(
            auto_heal=True,
            confidence_threshold=0.6,
            update_on_heal=True,
            max_healing_attempts=10,
            learn_successful_strategies=True,
        )
        return cls(config)

    @classmethod
    def create_conservative(cls) -> HealingManager:
        """Create healing manager with conservative healing settings.

        Conservative mode:
        - Auto-heal enabled
        - Does not update patterns
        - Higher confidence threshold
        - Fewer fallback strategies

        Returns:
            HealingManager with conservative settings
        """
        config = HealingConfig(
            auto_heal=True,
            confidence_threshold=0.85,
            update_on_heal=False,
            max_healing_attempts=3,
            learn_successful_strategies=True,
            fallback_strategies=["visual", "text"],
        )
        return cls(config)

    def __str__(self) -> str:
        """String representation.

        Returns:
            Human-readable description
        """
        return (
            f"HealingManager(auto_heal={self.config.auto_heal}, "
            f"threshold={self.config.confidence_threshold}, "
            f"strategies={len(self.locator.strategies)})"
        )

    def __repr__(self) -> str:
        """Developer representation.

        Returns:
            Detailed description
        """
        stats = self.get_healing_stats()
        return (
            f"HealingManager(config={self.config}, "
            f"healing_attempts={stats['total_attempts']}, "
            f"healing_rate={stats['healing_rate']:.2f})"
        )
