"""Template matching engine for image recognition.

Handles finding patterns on screen using FindAction with strategy-based
result selection.
"""

import asyncio
import logging
from typing import Any

from ..actions.action_result import ActionResult, ActionResultBuilder
from ..actions.find import FindAction, FindOptions
from ..find.match import Match as FindMatch
from ..model.element import Pattern

logger = logging.getLogger(__name__)


class TemplateMatchEngine:
    """Finds patterns on screen using template matching.

    Responsibilities:
    - Execute pattern finding via FindAction
    - Manage async pattern search for multiple patterns
    - Apply search strategies (FIRST/BEST/ALL/EACH)
    - Convert FindResult to ActionResult
    - Cache results in context
    """

    def __init__(self, context: Any, pattern_loader: Any):
        """Initialize engine with context and pattern loader.

        Args:
            context: Execution context for result caching
            pattern_loader: PatternLoader for loading patterns from config
        """
        self.context = context
        self.pattern_loader = pattern_loader

    def find_images(
        self, image_ids: list[str], similarity: float, strategy: str = "FIRST"
    ) -> ActionResult | None:
        """Find images on screen by loading patterns and matching.

        Args:
            image_ids: List of image IDs to find
            similarity: Similarity threshold (0.0-1.0)
            strategy: Search strategy (FIRST, BEST, ALL, EACH)

        Returns:
            ActionResult with matches or None if not found
        """
        import os
        import tempfile

        from qontinui_schemas.common import utc_now

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(tempfile.gettempdir(), "qontinui_find_debug.log")
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] TEMPLATE_MATCH_ENGINE: {msg}\n")
            except Exception:
                pass

        log_debug("find_images() called")
        log_debug(f"  image_ids: {image_ids}")
        log_debug(f"  similarity: {similarity}")
        log_debug(f"  strategy: {strategy}")

        # Load patterns from config
        log_debug("  Loading patterns from config...")
        patterns = self.pattern_loader.load_patterns(image_ids)
        log_debug(f"  Loaded {len(patterns) if patterns else 0} patterns")

        if not patterns:
            log_debug("  ERROR: No patterns loaded!")
            logger.error(f"Could not load patterns for image IDs: {image_ids}")
            return None

        logger.debug(
            f"Finding {len(patterns)} patterns with threshold={similarity}, strategy={strategy}"
        )

        log_debug("  Calling find_patterns()...")
        # Find patterns on screen
        result = self.find_patterns(
            patterns=patterns,
            threshold=similarity,
            image_ids=image_ids,
            strategy=strategy,
        )
        log_debug(f"  find_patterns() returned: {result}")
        return result

    def find_patterns(
        self,
        patterns: list[Pattern],
        threshold: float,
        image_ids: list[str],
        strategy: str = "FIRST",
    ) -> ActionResult | None:
        """Find multiple patterns on screen with strategy-based selection.

        Args:
            patterns: List of Pattern objects to find
            threshold: Similarity threshold (0.0-1.0)
            image_ids: List of image IDs for metadata tagging
            strategy: Search strategy (FIRST, BEST, ALL, EACH)

        Returns:
            ActionResult with matches or None if not found
        """
        # Single pattern: use synchronous path
        if len(patterns) == 1:
            return self._find_single_pattern(
                pattern=patterns[0],
                threshold=threshold,
                image_id=image_ids[0] if image_ids else None,
            )

        # Multiple patterns: use async execution
        logger.debug(f"Finding {len(patterns)} patterns with strategy={strategy}")

        async_results = self._run_async_find(patterns, threshold, image_ids)

        if not async_results:
            return None

        # Build ActionResult based on strategy
        action_result = self._build_result_with_strategy(
            async_results=async_results, strategy=strategy, image_ids=image_ids
        )

        if action_result:
            self.context.update_last_action_result(action_result)
            logger.debug(f"Found {len(action_result.matches)} matches using {strategy} strategy")

        return action_result

    def _find_single_pattern(
        self, pattern: Pattern, threshold: float, image_id: str | None = None
    ) -> ActionResult | None:
        """Find single pattern on screen synchronously.

        Args:
            pattern: Pattern to find
            threshold: Similarity threshold (0.0-1.0)
            image_id: Optional image ID for metadata tagging

        Returns:
            ActionResult with matches or None if not found
        """
        import os
        import tempfile

        from qontinui_schemas.common import utc_now

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(tempfile.gettempdir(), "qontinui_find_debug.log")
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] TEMPLATE_MATCH_ENGINE: {msg}\n")
            except Exception:
                pass

        log_debug("_find_single_pattern() called")
        log_debug(f"  pattern.name: {pattern.name}")
        log_debug(f"  threshold: {threshold}")
        log_debug(f"  image_id: {image_id}")

        logger.debug(f"Finding pattern: {pattern.name} (threshold={threshold})")

        # Use FindAction for template matching
        log_debug("  Creating FindAction instance...")
        action = FindAction()

        # Get monitor_index - first try from StateImage config, then fall back to context
        monitor_index = None
        if image_id:
            monitors = self.pattern_loader.get_monitors_for_image(image_id)
            if monitors:
                monitor_index = monitors[0]  # Use first monitor from list
                log_debug(f"  Using monitor from StateImage config: {monitor_index}")

        # Fall back to context monitor_index if not set from StateImage
        if monitor_index is None:
            monitor_index = getattr(self.context, "monitor_index", None)
            log_debug(f"  Using monitor_index from context: {monitor_index}")

        log_debug("  Calling FindAction.find()...")
        find_result = action.find(
            pattern=pattern,
            options=FindOptions(similarity=threshold, find_all=False, monitor_index=monitor_index),
        )

        log_debug(f"  FindAction.find() returned: {find_result}")
        log_debug(f"  find_result.found: {find_result.found if find_result else 'N/A'}")
        log_debug(f"  find_result.matches: {find_result.matches if find_result else 'N/A'}")

        if not find_result.found or not find_result.matches:
            log_debug(f"  Pattern {pattern.name} not found - returning None")
            logger.debug(f"Pattern {pattern.name} not found")
            return None

        log_debug("  Pattern FOUND! Building ActionResult...")
        # Convert to ActionResult
        builder = ActionResultBuilder().with_success(True)

        for i, model_match in enumerate(find_result.matches):
            log_debug(f"    Adding match {i}: {model_match}")
            find_match = FindMatch(match_object=model_match)
            builder.add_match(find_match)

            # Tag with source image ID
            if image_id and not hasattr(model_match.metadata, "source_image_id"):
                model_match.metadata.source_image_id = image_id
                log_debug(f"      Tagged with source_image_id: {image_id}")

        action_result = builder.build()
        log_debug(f"  ActionResult built: {action_result}")
        log_debug(f"  ActionResult.success: {action_result.success}")
        log_debug(f"  ActionResult.matches count: {len(action_result.matches)}")

        self.context.update_last_action_result(action_result)
        log_debug("  Updated context.last_action_result")

        logger.debug(f"Found {len(action_result.matches)} matches for pattern {pattern.name}")
        return action_result

    def _run_async_find(
        self, patterns: list[Pattern], threshold: float, image_ids: list[str] | None = None
    ) -> list[tuple[Pattern, int, Any]]:
        """Run async pattern finding for multiple patterns.

        Args:
            patterns: List of patterns to find
            threshold: Similarity threshold (0.0-1.0)
            image_ids: Optional list of image IDs for per-pattern monitor lookup

        Returns:
            List of (pattern, index, find_result) tuples
        """
        # Get default monitor_index from context
        default_monitor_index = getattr(self.context, "monitor_index", None)

        # Build per-pattern monitor indices
        pattern_monitors: list[int | None] = []
        for i in range(len(patterns)):
            monitor_index = None
            if image_ids and i < len(image_ids):
                monitors = self.pattern_loader.get_monitors_for_image(image_ids[i])
                if monitors:
                    monitor_index = monitors[0]
            if monitor_index is None:
                monitor_index = default_monitor_index
            pattern_monitors.append(monitor_index)

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def find_all_patterns():
            """Find all patterns using async parallelism."""
            action = FindAction()
            tasks = []

            for i, pattern in enumerate(patterns):
                task = asyncio.create_task(
                    asyncio.to_thread(
                        action.find,
                        pattern=pattern,
                        options=FindOptions(
                            similarity=threshold,
                            find_all=False,
                            monitor_index=pattern_monitors[i],
                        ),
                    )
                )
                tasks.append((pattern, i, task))

            # Gather all results
            results = []
            for pattern, idx, task in tasks:
                try:
                    result = await task
                    if result.found and result.matches:
                        results.append((pattern, idx, result))
                except Exception as e:
                    logger.warning(f"Pattern {pattern.name} search failed: {e}")
                    continue

            return results

        return loop.run_until_complete(find_all_patterns())

    def _build_result_with_strategy(
        self,
        async_results: list[tuple[Pattern, int, Any]],
        strategy: str,
        image_ids: list[str],
    ) -> ActionResult | None:
        """Build ActionResult by applying search strategy.

        Args:
            async_results: List of (pattern, index, find_result) tuples
            strategy: Search strategy (FIRST, BEST, ALL, EACH)
            image_ids: List of image IDs for metadata tagging

        Returns:
            ActionResult with strategy-filtered matches or None
        """
        if not async_results:
            return None

        builder = ActionResultBuilder().with_success(True)

        if strategy == "FIRST":
            # Use first match found
            _, idx, find_result = async_results[0]
            self._add_matches_to_builder(builder, find_result, idx, image_ids)

        elif strategy == "BEST":
            # Find result with highest confidence
            best_tuple = max(
                async_results,
                key=lambda r: r[2].matches[0].score if r[2].matches else 0,
            )
            _, idx, find_result = best_tuple
            self._add_matches_to_builder(builder, find_result, idx, image_ids)

        elif strategy == "ALL":
            # Collect ALL matches from ALL patterns
            for _, idx, find_result in async_results:
                self._add_matches_to_builder(builder, find_result, idx, image_ids)

        elif strategy == "EACH":
            # One best match per pattern
            for _, idx, find_result in async_results:
                if find_result.matches:
                    # Take only the best match from this pattern
                    model_match = find_result.matches[0]
                    find_match = FindMatch(match_object=model_match)
                    builder.add_match(find_match)

                    # Tag with source image ID
                    if idx < len(image_ids):
                        model_match.metadata.source_image_id = image_ids[idx]

        else:
            logger.error(f"Unknown strategy: {strategy}")
            return None

        return builder.build()

    def _add_matches_to_builder(
        self,
        builder: ActionResultBuilder,
        find_result: Any,
        idx: int,
        image_ids: list[str],
    ) -> None:
        """Add matches from find_result to builder with metadata tagging.

        Args:
            builder: ActionResultBuilder to add matches to
            find_result: FindResult with matches
            idx: Index in image_ids list
            image_ids: List of image IDs for metadata tagging
        """
        for model_match in find_result.matches:
            find_match = FindMatch(match_object=model_match)
            builder.add_match(find_match)

            # Tag with source image ID
            if idx < len(image_ids):
                model_match.metadata.source_image_id = image_ids[idx]
