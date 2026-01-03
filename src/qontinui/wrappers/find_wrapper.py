"""FindWrapper - Routes pattern finding to the consolidated FindAction system.

This wrapper provides the entry point for pattern finding operations,
delegating to FindAction which handles mock/real routing internally.

Architecture:
    High-level APIs (StateDetector, etc.)
      ↓
    FindWrapper (this layer) ← Entry point
      ↓
    FindAction → FindWrapper (actions/find/) → Mock or Real implementation
      ↓
    FindResult (unified return type)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..actions.find import FindAction, FindOptions
from ..actions.find.find_result import FindResult
from ..model.element.region import Region
from .base import BaseWrapper

if TYPE_CHECKING:
    from ..model.element.pattern import Pattern
    from ..model.match.match import Match

logger = logging.getLogger(__name__)


class FindWrapper(BaseWrapper):
    """Wrapper for pattern finding operations.

    Delegates all find operations to the consolidated FindAction system.
    The FindAction handles mock/real routing internally, so this wrapper
    just provides a convenient API.

    All methods return FindResult for consistency.
    """

    def __init__(self) -> None:
        """Initialize FindWrapper."""
        super().__init__()
        self._find_action: FindAction | None = None
        logger.debug("FindWrapper initialized")

    @property
    def find_action(self) -> FindAction:
        """Get FindAction instance (lazy initialization)."""
        if self._find_action is None:
            self._find_action = FindAction()
            logger.debug("FindAction initialized")
        return self._find_action

    async def find(
        self,
        pattern: Pattern,
        search_region: Region | None = None,
        similarity: float | None = None,
    ) -> FindResult:
        """Find a pattern (single best match).

        Delegates to FindAction which handles mock/real routing.

        Args:
            pattern: Pattern to find
            search_region: Optional region to search in
            similarity: Optional similarity threshold override

        Returns:
            FindResult with match data (works identically for mock and real)
        """
        options = FindOptions(
            search_region=search_region,
            find_all=False,
        )
        if similarity is not None:
            options = FindOptions(
                similarity=similarity,
                search_region=search_region,
                find_all=False,
            )

        logger.debug(f"FindWrapper.find: {pattern.name}")
        return await self.find_action.find(pattern, options)

    async def find_all(
        self,
        pattern: Pattern,
        search_region: Region | None = None,
        similarity: float | None = None,
    ) -> list[Match]:
        """Find all occurrences of a pattern.

        Delegates to FindAction which handles mock/real routing.

        Args:
            pattern: Pattern to find
            search_region: Optional region to search in
            similarity: Optional similarity threshold override

        Returns:
            List of Match objects (empty if none found)
        """
        options = FindOptions(
            search_region=search_region,
            find_all=True,
        )
        if similarity is not None:
            options = FindOptions(
                similarity=similarity,
                search_region=search_region,
                find_all=True,
            )

        logger.debug(f"FindWrapper.find_all: {pattern.name}")
        result = await self.find_action.find(pattern, options)
        return result.matches.to_list()

    async def wait_for(
        self,
        pattern: Pattern,
        timeout: float = 5.0,
        search_region: Region | None = None,
        similarity: float | None = None,
    ) -> Match | None:
        """Wait for a pattern to appear.

        Polls for pattern at regular intervals until found or timeout.

        Args:
            pattern: Pattern to wait for
            timeout: Maximum wait time in seconds
            search_region: Optional region to search in
            similarity: Optional similarity threshold override

        Returns:
            Match if found within timeout, None otherwise
        """
        import asyncio
        import time

        options = FindOptions(
            search_region=search_region,
            find_all=False,
        )
        if similarity is not None:
            options = FindOptions(
                similarity=similarity,
                search_region=search_region,
                find_all=False,
            )

        logger.debug(f"FindWrapper.wait_for: {pattern.name}, timeout={timeout}")

        start_time = time.time()
        poll_interval = 0.5

        while time.time() - start_time < timeout:
            result = await self.find_action.find(pattern, options)
            if result.found and result.best_match:
                return result.best_match
            await asyncio.sleep(poll_interval)

        logger.debug(f"Timeout waiting for pattern {pattern.name} after {timeout}s")
        return None

    async def exists(
        self,
        pattern: Pattern,
        search_region: Region | None = None,
        similarity: float | None = None,
    ) -> bool:
        """Check if a pattern exists on screen.

        Args:
            pattern: Pattern to check for
            search_region: Optional region to search in
            similarity: Optional similarity threshold override

        Returns:
            True if pattern is found, False otherwise
        """
        result = await self.find(pattern, search_region, similarity)
        return result.found

    async def find_patterns(
        self,
        patterns: list[Pattern],
        search_region: Region | None = None,
        similarity: float | None = None,
        max_concurrent: int = 15,
    ) -> list[FindResult]:
        """Find multiple patterns asynchronously.

        Args:
            patterns: List of patterns to find
            search_region: Optional region to search in
            similarity: Optional similarity threshold override
            max_concurrent: Maximum concurrent pattern searches

        Returns:
            List of FindResults, one per pattern
        """
        options = FindOptions(
            search_region=search_region,
            find_all=False,
        )
        if similarity is not None:
            options = FindOptions(
                similarity=similarity,
                search_region=search_region,
                find_all=False,
            )

        logger.debug(f"FindWrapper.find_patterns: {len(patterns)} patterns")
        return await self.find_action.find(patterns, options, max_concurrent)
