"""MockFindImplementation - Returns pre-recorded matches.

This is the mock execution implementation in model-based GUI automation.
It returns matches from ActionHistory without any actual image finding.
"""

import logging
import random

from ...model.element import Location, Pattern, Region
from ...model.match import Match
from .find_options import FindOptions
from .find_result import FindResult

logger = logging.getLogger(__name__)


class MockFindImplementation:
    """Returns pre-recorded matches from ActionHistory.

    Single Responsibility: Provide mock find execution using historical data.

    This implementation returns FindResult in the SAME format as
    RealFindImplementation, so the action layer is completely agnostic.
    """

    def execute(
        self,
        pattern: Pattern,
        options: FindOptions,
    ) -> FindResult:
        """Execute mock find operation.

        Returns matches from ActionHistory or generates mock matches.

        Args:
            pattern: Pattern to find
            options: Find configuration

        Returns:
            FindResult with historical or generated matches
        """
        # Try to get matches from ActionHistory
        if hasattr(pattern, "match_history") and not pattern.match_history.is_empty():
            matches = self._get_matches_from_history(pattern, options)
        else:
            matches = self._generate_mock_matches(pattern, options)

        # Filter by find_all option
        if not options.find_all and matches:
            matches = [matches[0]]

        # Mock timing
        duration_ms = random.uniform(10.0, 50.0)

        return FindResult(
            matches=matches,
            found=len(matches) > 0,
            pattern_name=pattern.name,
            duration_ms=duration_ms,
            debug_data=None,  # No debug data in mock mode
        )

    def _get_matches_from_history(
        self,
        pattern: Pattern,
        options: FindOptions,
    ) -> list[Match]:
        """Get matches from pattern's ActionHistory."""
        # Get random snapshot from history
        snapshot = pattern.match_history.get_random_snapshot(
            active_states=set(), action_type="FIND"  # TODO: integrate with state management
        )

        if snapshot and snapshot.matches:
            return snapshot.matches

        return []

    def _generate_mock_matches(
        self,
        pattern: Pattern,
        options: FindOptions,
    ) -> list[Match]:
        """Generate mock matches when no history available."""
        # Generate random match within search region
        region = options.search_region or Region(0, 0, 1920, 1080)

        x = region.x + random.randint(0, max(0, region.width - 100))
        y = region.y + random.randint(0, max(0, region.height - 50))

        match = Match(
            target=Location(region=Region(x, y, 100, 50)),
            score=random.uniform(0.85, 0.99),
            ocr_text="",
        )

        return [match]

    async def execute_async(
        self,
        patterns: list[Pattern],
        options: FindOptions,
        max_concurrent: int = 15,
    ) -> list[FindResult]:
        """Execute async mock find operations for multiple patterns.

        Args:
            patterns: List of patterns to find
            options: Find configuration
            max_concurrent: Maximum concurrent pattern searches (ignored in mock)

        Returns:
            List of FindResults for each pattern
        """
        import asyncio

        logger.info(f"[MOCK] Executing async find for {len(patterns)} patterns")

        # Execute synchronously in mock mode (fast anyway)
        # But wrap in asyncio.to_thread to simulate async behavior
        tasks = [asyncio.to_thread(self.execute, pattern, options) for pattern in patterns]

        results = await asyncio.gather(*tasks)
        return results
