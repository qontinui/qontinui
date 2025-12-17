"""Multi-strategy locator that tries multiple finding methods.

Orchestrates multiple locator strategies to find elements, trying each in sequence
until one succeeds. This reduces test brittleness by providing fallback options.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..model.element import Pattern
from .strategies import (
    ColorRegionStrategy,
    LocatorStrategy,
    MatchResult,
    RelativePositionStrategy,
    ScreenContext,
    SemanticTextStrategy,
    StructuralStrategy,
    VisualPatternStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class LocatorAttempt:
    """Record of a locator strategy attempt.

    Attributes:
        strategy_name: Name of strategy that was tried
        success: Whether strategy found a match
        confidence: Match confidence if successful
        duration: Time taken by strategy
        error: Error message if strategy failed
    """

    strategy_name: str
    success: bool
    confidence: float = 0.0
    duration: float = 0.0
    error: str | None = None


@dataclass
class MultiStrategyResult:
    """Result from multi-strategy locator.

    Attributes:
        match_result: Match result if found
        successful_strategy: Name of strategy that succeeded
        attempts: List of all strategy attempts
        total_duration: Total time taken
        metadata: Additional result metadata
    """

    match_result: MatchResult | None
    successful_strategy: str | None
    attempts: list[LocatorAttempt] = field(default_factory=list)
    total_duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        """Check if element was found.

        Returns:
            True if match_result is not None
        """
        return self.match_result is not None

    @property
    def confidence(self) -> float:
        """Get match confidence.

        Returns:
            Confidence score or 0.0 if not found
        """
        return self.match_result.confidence if self.match_result else 0.0


class MultiStrategyLocator:
    """Locator that tries multiple strategies in sequence.

    Reduces test brittleness by providing multiple ways to find the same element.
    Strategies are tried in order until one succeeds.

    Example:
        >>> locator = MultiStrategyLocator()
        >>> locator.add_strategy(VisualPatternStrategy())  # Primary
        >>> locator.add_strategy(SemanticTextStrategy())   # Fallback 1
        >>> locator.add_strategy(RelativePositionStrategy())  # Fallback 2
        >>>
        >>> result = locator.find(pattern, context, min_confidence=0.8)
        >>> if result.found:
        ...     print(f"Found using {result.successful_strategy}")
    """

    def __init__(self, strategies: list[LocatorStrategy] | None = None) -> None:
        """Initialize multi-strategy locator.

        Args:
            strategies: List of strategies to use (default: [VisualPatternStrategy])
        """
        if strategies is None:
            # Default to visual pattern matching only
            self.strategies: list[LocatorStrategy] = [VisualPatternStrategy()]
        else:
            self.strategies = strategies

        self._attempt_history: list[LocatorAttempt] = []

    def add_strategy(self, strategy: LocatorStrategy) -> None:
        """Add a locator strategy.

        Args:
            strategy: Strategy to add to the list
        """
        self.strategies.append(strategy)

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy by name.

        Args:
            strategy_name: Name of strategy to remove

        Returns:
            True if strategy was removed, False if not found
        """
        initial_count = len(self.strategies)
        self.strategies = [s for s in self.strategies if s.get_name() != strategy_name]
        return len(self.strategies) < initial_count

    def clear_strategies(self) -> None:
        """Remove all strategies."""
        self.strategies.clear()

    def find(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
        stop_on_first: bool = True,
    ) -> MultiStrategyResult:
        """Find element using multiple strategies.

        Tries each strategy in sequence until one succeeds or all fail.
        Logs which strategy succeeded for learning purposes.

        Args:
            target: Pattern or configuration dict describing what to find
            context: Screen context with screenshot
            min_confidence: Minimum confidence threshold
            stop_on_first: Stop after first successful match (default True)

        Returns:
            MultiStrategyResult with match and attempt history
        """
        start_time = time.time()
        attempts: list[LocatorAttempt] = []
        best_result: MatchResult | None = None
        successful_strategy: str | None = None

        logger.debug(
            f"Multi-strategy search starting with {len(self.strategies)} strategies "
            f"(min_confidence={min_confidence})"
        )

        for strategy in self.strategies:
            # Check if strategy can handle this target
            if not strategy.can_handle(target):
                logger.debug(f"Strategy {strategy.get_name()} cannot handle target type")
                continue

            strategy_start = time.time()
            attempt = LocatorAttempt(strategy_name=strategy.get_name(), success=False)

            try:
                logger.debug(f"Trying strategy: {strategy.get_name()}")
                result = strategy.find(target, context, min_confidence)
                strategy_duration = time.time() - strategy_start

                attempt.duration = strategy_duration

                if result and result.confidence >= min_confidence:
                    attempt.success = True
                    attempt.confidence = result.confidence

                    logger.info(
                        f"Strategy {strategy.get_name()} succeeded "
                        f"(confidence={result.confidence:.3f}, duration={strategy_duration:.3f}s)"
                    )

                    # Keep track of best result
                    if best_result is None or result.confidence > best_result.confidence:
                        best_result = result
                        successful_strategy = strategy.get_name()

                    if stop_on_first:
                        attempts.append(attempt)
                        break
                else:
                    logger.debug(
                        f"Strategy {strategy.get_name()} failed "
                        f"(confidence={result.confidence if result else 0.0:.3f})"
                    )

            except Exception as e:
                strategy_duration = time.time() - strategy_start
                attempt.duration = strategy_duration
                attempt.error = str(e)
                logger.error(
                    f"Strategy {strategy.get_name()} error: {e}",
                    exc_info=True,
                )

            attempts.append(attempt)

        total_duration = time.time() - start_time

        # Record attempt in history
        self._attempt_history.extend(attempts)

        # Log summary
        if best_result:
            logger.info(
                f"Multi-strategy search succeeded using {successful_strategy} "
                f"(total_duration={total_duration:.3f}s, attempts={len(attempts)})"
            )
        else:
            logger.warning(
                f"Multi-strategy search failed after {len(attempts)} attempts "
                f"(total_duration={total_duration:.3f}s)"
            )

        return MultiStrategyResult(
            match_result=best_result,
            successful_strategy=successful_strategy,
            attempts=attempts,
            total_duration=total_duration,
            metadata={
                "strategies_available": len(self.strategies),
                "strategies_tried": len(attempts),
                "min_confidence": min_confidence,
            },
        )

    def find_all_strategies(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
    ) -> list[MultiStrategyResult]:
        """Try all strategies and return all results.

        Unlike find(), this tries all strategies even if one succeeds.
        Useful for comparing strategy effectiveness.

        Args:
            target: Pattern or configuration dict
            context: Screen context
            min_confidence: Minimum confidence threshold

        Returns:
            List of results, one per strategy
        """
        return [self.find(target, context, min_confidence, stop_on_first=False)]

    def get_attempt_history(self) -> list[LocatorAttempt]:
        """Get history of all locator attempts.

        Returns:
            List of all attempts across all find() calls
        """
        return self._attempt_history.copy()

    def clear_history(self) -> None:
        """Clear attempt history."""
        self._attempt_history.clear()

    def get_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Get success statistics for each strategy.

        Returns:
            Dict mapping strategy names to stats (attempts, successes, avg_confidence)
        """
        stats: dict[str, dict[str, Any]] = {}

        for attempt in self._attempt_history:
            strategy_name = attempt.strategy_name

            if strategy_name not in stats:
                stats[strategy_name] = {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_confidence": 0.0,
                    "total_duration": 0.0,
                }

            stats[strategy_name]["attempts"] += 1
            stats[strategy_name]["total_duration"] += attempt.duration

            if attempt.success:
                stats[strategy_name]["successes"] += 1
                stats[strategy_name]["total_confidence"] += attempt.confidence
            else:
                stats[strategy_name]["failures"] += 1

        # Calculate averages
        for _strategy_name, strategy_stats in stats.items():
            attempts = strategy_stats["attempts"]
            successes = strategy_stats["successes"]

            strategy_stats["success_rate"] = (
                float(successes) / float(attempts) if attempts > 0 else 0.0
            )
            strategy_stats["avg_confidence"] = (
                strategy_stats["total_confidence"] / float(successes) if successes > 0 else 0.0
            )
            strategy_stats["avg_duration"] = (
                strategy_stats["total_duration"] / float(attempts) if attempts > 0 else 0.0
            )

        return stats

    @classmethod
    def create_default(cls) -> MultiStrategyLocator:
        """Create locator with default strategy set.

        Returns:
            MultiStrategyLocator with standard strategies
        """
        return cls(
            strategies=[
                VisualPatternStrategy(),  # Primary
                SemanticTextStrategy(),  # Fallback 1
                RelativePositionStrategy(),  # Fallback 2
                ColorRegionStrategy(),  # Fallback 3
                StructuralStrategy(),  # Fallback 4
            ]
        )

    @classmethod
    def create_visual_only(cls) -> MultiStrategyLocator:
        """Create locator with only visual pattern matching.

        Returns:
            MultiStrategyLocator with single visual strategy
        """
        return cls(strategies=[VisualPatternStrategy()])

    @classmethod
    def create_with_strategies(cls, *strategy_names: str) -> MultiStrategyLocator:
        """Create locator with specified strategies.

        Args:
            *strategy_names: Names of strategies to include
                           (visual, text, relative, color, structural)

        Returns:
            MultiStrategyLocator with requested strategies

        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_map = {
            "visual": VisualPatternStrategy,
            "text": SemanticTextStrategy,
            "relative": RelativePositionStrategy,
            "color": ColorRegionStrategy,
            "structural": StructuralStrategy,
        }

        strategies = []
        for name in strategy_names:
            name_lower = name.lower()
            if name_lower not in strategy_map:
                raise ValueError(
                    f"Unknown strategy: {name}. " f"Available: {', '.join(strategy_map.keys())}"
                )
            strategies.append(strategy_map[name_lower]())  # type: ignore[abstract]

        return cls(strategies=strategies)

    def __str__(self) -> str:
        """String representation.

        Returns:
            Human-readable description
        """
        strategy_names = [s.get_name() for s in self.strategies]
        return f"MultiStrategyLocator(strategies={strategy_names})"

    def __repr__(self) -> str:
        """Developer representation.

        Returns:
            Detailed description
        """
        return (
            f"MultiStrategyLocator(strategies={len(self.strategies)}, "
            f"attempts_history={len(self._attempt_history)})"
        )
