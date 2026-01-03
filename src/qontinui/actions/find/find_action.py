"""FindAction - Single entry point for all image finding operations.

This is THE action class for finding images in qontinui. All find operations
go through this class, whether from FIND actions, CLICK actions, IF conditions,
or state verification.

Following model-based GUI automation:
- High-level action class (this)
- Wrapper for mock/real delegation (FindWrapper)
- Implementation classes (MockFindImplementation, RealFindImplementation)
"""

from typing import cast, overload

from ...model.element import Pattern
from .find_options import FindOptions
from .find_result import FindResult


class FindAction:
    """Single entry point for all image finding operations.

    This action class is agnostic to whether execution is mocked or real.
    It delegates to FindWrapper which handles the mock/real decision.

    All find operations use the cascade pattern for options:
    1. Explicit overrides (via build_find_options)
    2. SearchOptions (from JSON config)
    3. Pattern-level overrides
    4. StateImage config
    5. Project config (QontinuiSettings)
    6. Library defaults (action_defaults)

    Usage:
        # Single pattern
        from qontinui.actions.find import FindAction
        from qontinui.model.element import Pattern

        pattern = Pattern.from_file("button.png")
        action = FindAction()
        result = await action.find(pattern)  # Returns FindResult

        # Multiple patterns (parallel)
        patterns = [Pattern.from_file(f"icon{i}.png") for i in range(5)]
        results = await action.find(patterns)  # Returns list[FindResult]

        # With options from cascade
        from qontinui.actions.find.find_options_builder import CascadeContext, build_find_options

        ctx = CascadeContext(
            search_options=action_config.search_options,
            pattern=pattern,
            project_config=QontinuiSettings(),
        )
        options = build_find_options(ctx)
        result = await action.find(pattern, options)
    """

    def __init__(self):
        """Initialize FindAction with wrapper."""
        from .find_wrapper import FindWrapper

        self._wrapper = FindWrapper()

    @overload
    async def find(
        self,
        pattern: Pattern,
        options: FindOptions | None = None,
        max_concurrent: int = 15,
    ) -> FindResult: ...

    @overload
    async def find(
        self,
        pattern: list[Pattern],
        options: FindOptions | None = None,
        max_concurrent: int = 15,
    ) -> list[FindResult]: ...

    async def find(
        self,
        pattern: Pattern | list[Pattern],
        options: FindOptions | None = None,
        max_concurrent: int = 15,
    ) -> FindResult | list[FindResult]:
        """Find image(s) on screen.

        Args:
            pattern: Pattern or list of patterns to find
            options: Find configuration (uses defaults if None)
            max_concurrent: Maximum concurrent pattern searches

        Returns:
            FindResult for single pattern, list[FindResult] for multiple
        """
        options = options or FindOptions()

        # Handle single pattern
        if isinstance(pattern, Pattern):
            results = await self._wrapper.find([pattern], options, max_concurrent)
            return cast(FindResult, results[0])

        # Handle multiple patterns
        return cast(list[FindResult], await self._wrapper.find(pattern, options, max_concurrent))
