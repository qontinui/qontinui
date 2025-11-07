"""FindAction - Single entry point for all image finding operations.

This is THE action class for finding images in qontinui. All find operations
go through this class, whether from FIND actions, CLICK actions, IF conditions,
or state verification.

Following model-based GUI automation:
- High-level action class (this)
- Wrapper for mock/real delegation (FindWrapper)
- Implementation classes (MockFindImplementation, RealFindImplementation)
"""

from ...model.element import Pattern
from .find_options import FindOptions
from .find_result import FindResult


class FindAction:
    """Single entry point for all image finding operations.

    This action class is agnostic to whether execution is mocked or real.
    It delegates to FindWrapper which handles the mock/real decision.

    Usage:
        # From FIND action
        action = FindAction()
        result = action.find(pattern, FindOptions(similarity=0.85))

        # From IF condition
        action = FindAction()
        result = action.find(pattern)
        if result.found:
            execute_then_branch()

        # From state verification
        action = FindAction()
        result = action.find(state_image_pattern)
        return result.found
    """

    def __init__(self):
        """Initialize FindAction with wrapper."""
        from .find_wrapper import FindWrapper

        self._wrapper = FindWrapper()

    def find(
        self,
        pattern: Pattern,
        options: FindOptions | None = None,
    ) -> FindResult:
        """Find an image on screen.

        Args:
            pattern: Pattern to find (includes image + optional mask)
            options: Find configuration (uses defaults if None)

        Returns:
            FindResult with matches (works identically for mock and real)
        """
        options = options or FindOptions()
        return self._wrapper.find(pattern, options)

    def exists(self, pattern: Pattern, similarity: float | None = None) -> bool:
        """Check if image exists (convenience for IF conditions).

        Args:
            pattern: Pattern to find
            similarity: Optional similarity threshold. If None, uses global default.

        Returns:
            True if pattern found, False otherwise
        """
        options = FindOptions(similarity=similarity) if similarity is not None else FindOptions()
        result = self.find(pattern, options)
        return result.found

    def wait_until_exists(
        self,
        pattern: Pattern,
        timeout: float = 10.0,
        similarity: float | None = None,
    ) -> FindResult:
        """Wait for image to appear (convenience for WAIT actions).

        Args:
            pattern: Pattern to find
            timeout: Maximum time to wait in seconds
            similarity: Optional similarity threshold. If None, uses global default.

        Returns:
            FindResult with best match if found, or empty result if timeout
        """
        options = (
            FindOptions(timeout=timeout, similarity=similarity)
            if similarity is not None
            else FindOptions(timeout=timeout)
        )
        result = self.find(pattern, options)
        return result

    async def find_async(
        self,
        patterns: list[Pattern],
        options: FindOptions | None = None,
        max_concurrent: int = 15,
    ) -> list[FindResult]:
        """Find multiple images asynchronously with parallel pattern matching.

        Args:
            patterns: List of patterns to find
            options: Find configuration (uses defaults if None)
            max_concurrent: Maximum concurrent pattern searches

        Returns:
            List of FindResults, one per pattern
        """
        options = options or FindOptions()
        return await self._wrapper.find_async(patterns, options, max_concurrent)

    async def exists_async(
        self,
        patterns: list[Pattern],
        similarity: float | None = None,
        max_concurrent: int = 15,
    ) -> dict[str, bool]:
        """Check if multiple images exist asynchronously.

        Args:
            patterns: List of patterns to check
            similarity: Optional similarity threshold. If None, uses global default.
            max_concurrent: Maximum concurrent pattern searches

        Returns:
            Dictionary mapping pattern names to existence (True/False)
        """
        options = FindOptions(similarity=similarity) if similarity is not None else FindOptions()
        results = await self.find_async(patterns, options, max_concurrent)

        # Convert results to dict of pattern name -> exists
        return {
            pattern.name: result.found for pattern, result in zip(patterns, results, strict=False)
        }
