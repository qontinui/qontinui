"""Async wrapper for image finding operations."""

import asyncio
import logging
from typing import Any, Callable

import numpy as np

from .......model.element.pattern import Pattern
from .......model.match.match import Match
from ....options.pattern_find_options import PatternFindOptions
from .search_executor import SearchExecutor

logger = logging.getLogger(__name__)


class AsyncFinder:
    """Provides async interface for image finding.

    Wraps synchronous template matching operations with async execution
    for improved performance when searching multiple patterns.

    Performance:
    - Sequential: N patterns × 200ms = N/5 seconds
    - Async parallel: ~200-400ms regardless of N
    """

    def __init__(self, max_concurrent: int = 15) -> None:
        """Initialize async finder.

        Args:
            max_concurrent: Maximum concurrent pattern searches
        """
        self.executor = SearchExecutor(max_concurrent)

    async def find_patterns(
        self,
        patterns: list[Pattern],
        capture_func: Callable[[], list[np.ndarray[Any, Any]]],
        search_func: Callable[[Pattern, list[np.ndarray[Any, Any]], PatternFindOptions], list[Match]],
        options: PatternFindOptions,
    ) -> list[Match]:
        """Find patterns asynchronously.

        Args:
            patterns: Patterns to search for
            capture_func: Function to capture search images
            search_func: Function to perform pattern search
            options: Pattern matching configuration

        Returns:
            All matches found across patterns
        """
        if not patterns:
            logger.warning("No patterns to find")
            return []

        # Capture search images once (shared across all searches)
        search_images = await asyncio.to_thread(capture_func)

        # Execute searches concurrently
        matches = await self.executor.execute_searches(
            patterns=patterns,
            search_images=search_images,
            search_func=search_func,
            options=options,
        )

        return matches
