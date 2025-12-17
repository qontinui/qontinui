"""FindWrapper - Routes find operations to mock or real implementation.

This is the delegation layer in model-based GUI automation architecture.
It's the ONLY class that knows about mock vs real execution.
"""

import logging

from ...config.framework_settings import FrameworkSettings
from ...model.element import Pattern
from .find_options import FindOptions
from .find_result import FindResult

logger = logging.getLogger(__name__)


class FindWrapper:
    """Routes find operations based on execution mode.

    This wrapper is the decision point for mock vs real execution.
    Both implementations return FindResult in the same format, so
    the action layer above is completely agnostic.

    Single Responsibility: Delegate to correct implementation.
    """

    def __init__(self):
        """Initialize wrapper with lazy-loaded implementations."""
        self._mock_impl = None
        self._real_impl = None

    @property
    def mock_implementation(self):
        """Get MockFindImplementation (lazy init)."""
        if self._mock_impl is None:
            from .mock_find_implementation import MockFindImplementation

            self._mock_impl = MockFindImplementation()
        return self._mock_impl

    @property
    def real_implementation(self):
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
        # File-based debug logging
        import os
        import tempfile
        from datetime import datetime

        debug_log = os.path.join(tempfile.gettempdir(), "qontinui_event_emission.log")
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] FindWrapper.find() ENTRY\n")
                f.write(
                    f"[{ts}]   pattern.id={pattern.id}, pattern.name={pattern.name}\n"
                )
        except Exception:
            pass

        logger.debug(
            f"[FIND_DEBUG] FindWrapper.find() ENTRY - pattern.id={pattern.id}, pattern.name={pattern.name}"
        )
        logger.debug(
            f"[FIND_DEBUG] Pattern pixel_data is None: {pattern.pixel_data is None}"
        )

        settings = FrameworkSettings.get_instance()
        is_mock = settings.core.mock

        logger.debug(f"[FIND_DEBUG] Execution mode: {'MOCK' if is_mock else 'REAL'}")

        # File-based debug logging for execution mode
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{ts}]   Execution mode: {'MOCK' if is_mock else 'REAL'}\n")
        except Exception:
            pass

        if is_mock:
            logger.debug(f"[MOCK] Finding pattern: {pattern.name}")
            result = self.mock_implementation.execute(pattern, options)
        else:
            logger.debug(f"[REAL] Finding pattern: {pattern.name}")
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}]   Calling real_implementation.execute()...\n")
            except Exception:
                pass
            result = self.real_implementation.execute(pattern, options)

        logger.debug(f"[FIND_DEBUG] FindWrapper.find() RETURN - found={result.found}")

        # File-based debug logging for result
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(
                    f"[{ts}]   FindWrapper.find() returning - found={result.found}\n"
                )
        except Exception:
            pass

        return result  # type: ignore[no-any-return]

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
