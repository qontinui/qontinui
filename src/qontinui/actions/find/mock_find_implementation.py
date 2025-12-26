"""MockFindImplementation - Returns pre-recorded matches.

This is the mock execution implementation in model-based GUI automation.
It returns matches from ActionHistory without any actual image finding.

The lookup order is:
1. Pattern's local ActionHistory (fastest, no API call)
2. API-based historical data from qontinui-api database (if enabled)
3. Generated mock based on state probability (fallback)
"""

import logging
import os
import random

from ...model.element import Location, Pattern, Region
from ...model.match import Match
from .find_options import FindOptions
from .find_result import FindResult
from .matches import Matches

logger = logging.getLogger(__name__)

# Flag to enable API-based historical data fetching
USE_API_HISTORICAL_DATA = os.getenv("QONTINUI_USE_API_HISTORICAL_DATA", "true").lower() == "true"


class MockFindImplementation:
    """Returns pre-recorded matches from ActionHistory.

    Single Responsibility: Provide mock find execution using historical data.

    This implementation returns FindResult in the SAME format as
    RealFindImplementation, so the action layer is completely agnostic.

    The lookup order is:
    1. Pattern's local ActionHistory (fastest, no API call)
    2. API-based historical data from qontinui-api database
    3. Generated mock based on state probability (fallback)
    """

    def __init__(self) -> None:
        """Initialize MockFindImplementation."""
        self._state_management = None

    @property
    def state_management(self):
        """Get MockStateManagement (lazy init)."""
        if self._state_management is None:
            try:
                from ...mock.mock_state_management import MockStateManagement

                self._state_management = MockStateManagement()
            except ImportError:
                logger.debug("MockStateManagement not available")
                self._state_management = None
        return self._state_management

    def execute(
        self,
        pattern: Pattern,
        options: FindOptions,
    ) -> FindResult:
        """Execute mock find operation.

        Returns matches from ActionHistory, API, or generates mock matches.

        Args:
            pattern: Pattern to find
            options: Find configuration

        Returns:
            FindResult with historical or generated matches
        """
        # Get active states for context
        active_states = self._get_active_states()

        # 1. Try to get matches from local ActionHistory
        if hasattr(pattern, "match_history") and not pattern.match_history.is_empty():
            matches = self._get_matches_from_history(pattern, active_states)
            if matches:
                return self._create_result(pattern, matches, options)

        # 2. Try to get historical data from API
        if USE_API_HISTORICAL_DATA:
            api_matches = self._get_matches_from_api(pattern, active_states)
            if api_matches:
                return self._create_result(pattern, api_matches, options)

        # 3. No history - generate mock based on state probability
        matches = self._generate_mock_matches(pattern, options, active_states)
        return self._create_result(pattern, matches, options)

    def _create_result(
        self,
        pattern: Pattern,
        matches: list[Match],
        options: FindOptions,
    ) -> FindResult:
        """Create FindResult from matches.

        Args:
            pattern: Pattern that was searched
            matches: List of matches found
            options: Find configuration

        Returns:
            FindResult with matches
        """
        # Filter by find_all option
        if not options.find_all and matches:
            matches = [matches[0]]

        # Mock timing
        duration_ms = random.uniform(10.0, 50.0)

        return FindResult(
            matches=Matches(matches),
            found=len(matches) > 0,
            pattern_name=pattern.name,
            duration_ms=duration_ms,
            debug_data=None,  # No debug data in mock mode
        )

    def _get_active_states(self) -> set[str]:
        """Get currently active states."""
        if self.state_management:
            return set(self.state_management.get_active_states())
        return set()

    def _get_matches_from_history(
        self,
        pattern: Pattern,
        active_states: set[str],
    ) -> list[Match]:
        """Get matches from pattern's ActionHistory.

        Args:
            pattern: Pattern with ActionHistory
            active_states: Currently active states

        Returns:
            List of matches from history, or empty list
        """
        # Get random snapshot preferring ones matching active states
        snapshot = pattern.match_history.get_random_snapshot(  # type: ignore[attr-defined]
            active_states=active_states,
            action_type="FIND",
        )

        if snapshot and hasattr(snapshot, "matches") and snapshot.matches:
            logger.debug(f"Using local historical snapshot with {len(snapshot.matches)} matches")
            return snapshot.matches  # type: ignore[no-any-return]

        if snapshot and hasattr(snapshot, "match_list") and snapshot.match_list:
            logger.debug(f"Using local historical snapshot with {len(snapshot.match_list)} matches")
            return snapshot.match_list  # type: ignore[no-any-return]

        return []

    def _get_matches_from_api(
        self,
        pattern: Pattern,
        active_states: set[str],
    ) -> list[Match]:
        """Get historical matches from qontinui-api.

        Args:
            pattern: Pattern to find historical data for
            active_states: Currently active states

        Returns:
            List of matches from API, or empty list
        """
        try:
            from ...mock.historical_data_client import get_historical_data_client

            client = get_historical_data_client()

            # Fetch random historical result matching pattern and context
            historical = client.get_random_historical_result(
                pattern_id=pattern.name,
                action_type="FIND",
                active_states=list(active_states) if active_states else None,
                success_only=False,
            )

            if not historical:
                return []

            if historical.success and historical.match_x is not None:
                match = Match(
                    target=Location(
                        region=Region(
                            historical.match_x,
                            historical.match_y or 0,
                            historical.match_width or 100,
                            historical.match_height or 50,
                        )
                    ),
                    score=historical.best_match_score or 0.9,
                    ocr_text="",
                )
                logger.debug(
                    f"Created match from API: ({historical.match_x}, {historical.match_y})"
                )
                return [match]

            return []

        except ImportError:
            logger.debug("Historical data client not available")
            return []
        except Exception as e:
            logger.warning(f"Error fetching historical data from API: {e}")
            return []

    def _generate_mock_matches(
        self,
        pattern: Pattern,
        options: FindOptions,
        active_states: set[str],
    ) -> list[Match]:
        """Generate mock matches when no history available.

        Uses state probability to determine if pattern should be found.

        Args:
            pattern: Pattern to mock
            options: Find configuration
            active_states: Currently active states

        Returns:
            List of mock matches (may be empty based on probability)
        """
        # Check if pattern's owner state is active
        if hasattr(pattern, "owner_state_name") and pattern.owner_state_name:
            if pattern.owner_state_name not in active_states and pattern.owner_state_name != "NULL":
                logger.debug(f"Pattern's state '{pattern.owner_state_name}' not active")
                return []

        # Use state probability to determine if found
        probability = 1.0
        if (
            self.state_management
            and hasattr(pattern, "owner_state_name")
            and pattern.owner_state_name
        ):
            if pattern.owner_state_name != "NULL":
                probability = (
                    self.state_management.get_state_probability(pattern.owner_state_name) / 100.0
                )

        if random.random() >= probability:
            logger.debug(f"Mock find failed (probability was {probability})")
            return []

        # Generate random match within search region
        region = options.search_region or Region(0, 0, 1920, 1080)

        x = region.x + random.randint(0, max(0, region.width - 100))
        y = region.y + random.randint(0, max(0, region.height - 50))
        width = random.randint(50, 200)
        height = random.randint(30, 100)

        match = Match(
            target=Location(region=Region(x, y, width, height)),
            score=random.uniform(0.85, 0.99),
            ocr_text="",
        )

        logger.debug(f"Generated mock match at ({x}, {y})")
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
