"""Concurrent search execution with semaphore control."""

import asyncio
import logging
from typing import Any, Callable

import numpy as np

from .......model.element.pattern import Pattern
from .......model.match.match import Match
from ....options.pattern_find_options import PatternFindOptions

logger = logging.getLogger(__name__)


class SearchExecutor:
    """Manages concurrent pattern search execution.

    Coordinates parallel template matching with configurable concurrency limits
    to prevent memory exhaustion while maximizing performance.
    """

    def __init__(self, max_concurrent: int = 15) -> None:
        """Initialize search executor.

        Args:
            max_concurrent: Maximum concurrent pattern searches
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_searches(
        self,
        patterns: list[Pattern],
        search_images: list[np.ndarray[Any, Any]],
        search_func: Callable[[Pattern, list[np.ndarray[Any, Any]], PatternFindOptions], list[Match]],
        options: PatternFindOptions,
    ) -> list[Match]:
        """Execute pattern searches concurrently.

        Args:
            patterns: Patterns to search for
            search_images: Images to search within
            search_func: Function that performs actual search
            options: Pattern matching configuration

        Returns:
            Combined matches from all pattern searches
        """
        logger.info(f"Starting parallel search for {len(patterns)} patterns")

        # Create search tasks
        tasks = [
            self._limited_search(pattern, search_images, search_func, options) for pattern in patterns
        ]

        # Execute all searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_matches = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during pattern search: {result}")
                continue

            all_matches.extend(result)

            # Early termination if needed
            if options.search_type.name == "FIRST" and all_matches:
                break

        logger.info(f"Parallel search complete: found {len(all_matches)} matches")
        return all_matches

    async def _limited_search(
        self,
        pattern: Pattern,
        search_images: list[np.ndarray[Any, Any]],
        search_func: Callable[[Pattern, list[np.ndarray[Any, Any]], PatternFindOptions], list[Match]],
        options: PatternFindOptions,
    ) -> list[Match]:
        """Execute single pattern search with concurrency limit.

        Args:
            pattern: Pattern to search for
            search_images: Images to search within
            search_func: Search function
            options: Pattern options

        Returns:
            Matches found for this pattern
        """
        async with self.semaphore:
            # Run CPU-bound template matching in thread pool
            matches = await asyncio.to_thread(search_func, pattern, search_images, options)
            return matches
