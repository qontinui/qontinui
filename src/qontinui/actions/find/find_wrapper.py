"""FindWrapper - Routes find operations to mock or real implementation.

This is the delegation layer in model-based GUI automation architecture.
It's the ONLY class that knows about mock vs real execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...config.framework_settings import FrameworkSettings
from ...model.element import Pattern
from .find_options import FindOptions
from .find_result import FindResult

if TYPE_CHECKING:
    from .mock_find_implementation import MockFindImplementation
    from .real_find_implementation import RealFindImplementation

logger = logging.getLogger(__name__)


class FindWrapper:
    """Routes find operations based on execution mode.

    This wrapper is the decision point for mock vs real execution.
    Both implementations return FindResult in the same format, so
    the action layer above is completely agnostic.

    Single Responsibility: Delegate to correct implementation.
    """

    def __init__(self) -> None:
        """Initialize wrapper with lazy-loaded implementations."""
        self._mock_impl: MockFindImplementation | None = None
        self._real_impl: RealFindImplementation | None = None

    @property
    def mock_implementation(self) -> MockFindImplementation:
        """Get MockFindImplementation (lazy init)."""
        if self._mock_impl is None:
            from .mock_find_implementation import MockFindImplementation

            self._mock_impl = MockFindImplementation()
        return self._mock_impl

    @property
    def real_implementation(self) -> RealFindImplementation:
        """Get RealFindImplementation (lazy init)."""
        if self._real_impl is None:
            from .real_find_implementation import RealFindImplementation

            self._real_impl = RealFindImplementation()
        return self._real_impl

    def find(
        self,
        pattern: Pattern,
        options: FindOptions,
    ) -> FindResult:
        """Find image using appropriate implementation.

        Checks ExecutionMode and routes to mock or real.

        Args:
            pattern: Pattern to find
            options: Find configuration

        Returns:
            FindResult (identical format from both implementations)
        """
        settings = FrameworkSettings.get_instance()
        is_mock = settings.core.mock

        logger.debug(
            f"FindWrapper.find: pattern={pattern.name}, mode={'MOCK' if is_mock else 'REAL'}"
        )

        if is_mock:
            result = self.mock_implementation.execute(pattern, options)
        else:
            result = self.real_implementation.execute(pattern, options)

        logger.debug(f"FindWrapper.find: found={result.found}, matches={len(result.matches)}")
        return result

    async def find_async(
        self,
        patterns: list[Pattern],
        options: FindOptions,
        max_concurrent: int = 15,
    ) -> list[FindResult]:
        """Find multiple images asynchronously using appropriate implementation.

        Checks ExecutionMode and routes to mock or real.

        Args:
            patterns: List of patterns to find
            options: Find configuration
            max_concurrent: Maximum concurrent pattern searches

        Returns:
            List of FindResults (identical format from both implementations)
        """
        settings = FrameworkSettings.get_instance()
        is_mock = settings.core.mock

        if is_mock:
            logger.debug(f"[MOCK] Finding {len(patterns)} patterns async")
            return await self.mock_implementation.execute_async(patterns, options, max_concurrent)  # type: ignore[no-any-return]
        else:
            logger.debug(f"[REAL] Finding {len(patterns)} patterns async")
            return await self.real_implementation.execute_async(patterns, options, max_concurrent)  # type: ignore[no-any-return]
