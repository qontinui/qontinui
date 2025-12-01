"""MockFind - Provides realistic pattern finding simulation using ActionHistory.

This module implements Brobot-style mocking where ActionHistory (containing ActionRecords)
drives the mock behavior, not manual probability settings.

For integration testing, MockFind can also fetch historical data from the qontinui-api
when the pattern's local ActionHistory is empty. This allows tests to use historical
data captured from real automation runs.
"""

import logging
import os
import random
from datetime import timedelta
from typing import cast

from ..actions.action_result import ActionResult, ActionResultBuilder
from ..mock.mock_mode_manager import MockModeManager
from ..mock.mock_state_management import MockStateManagement
from ..model.action import ActionRecord
from ..model.element import Location, Pattern, Region
from ..model.match import Match

logger = logging.getLogger(__name__)

# Flag to enable API-based historical data fetching
USE_API_HISTORICAL_DATA = (
    os.getenv("QONTINUI_USE_API_HISTORICAL_DATA", "true").lower() == "true"
)


class MockFind:
    """Simulates pattern finding operations using ActionHistory.

    Instead of manual probability configuration, MockFind uses ActionHistory
    attached to patterns to provide realistic mock behavior based on historical
    data or pre-configured snapshots.

    Key concepts:
    - Each Pattern has an ActionHistory with ActionRecords (snapshots)
    - ActionRecords contain matches found in specific state contexts
    - Mock finds return random snapshots matching current active states
    - If no history exists, falls back to generating simple mock matches
    """

    def __init__(self) -> None:
        """Initialize MockFind."""
        self.state_management = MockStateManagement()

    def find(
        self, pattern: Pattern, search_region: Region | None = None
    ) -> ActionResult:
        """Simulate finding a pattern using its ActionHistory.

        The lookup order is:
        1. Pattern's local ActionHistory (fastest, no API call)
        2. API-based historical data from qontinui-api database
        3. Generated mock based on state probability (fallback)

        Args:
            pattern: Pattern to find
            search_region: Optional region to search in

        Returns:
            ActionResult with matches from ActionHistory or generated mocks
        """
        MockModeManager.require_mock_mode()

        # Get currently active states
        active_states = set(self.state_management.get_active_states())
        logger.debug(
            f"MockFind: Finding pattern '{pattern.name}' with active states: {active_states}"
        )

        # 1. Try to get a snapshot from local ActionHistory
        snapshot = self._get_snapshot_from_history(pattern, active_states)

        if snapshot:
            # Use historical data from local ActionHistory
            logger.debug(
                f"Using local historical snapshot with {len(snapshot.match_list)} matches"
            )
            return self._create_result_from_snapshot(snapshot)

        # 2. Try to get historical data from API
        if USE_API_HISTORICAL_DATA:
            api_result = self._get_result_from_api(pattern, active_states)
            if api_result:
                logger.debug("Using API historical data")
                return api_result

        # 3. No history - generate mock based on state probability
        logger.debug(f"No history for pattern '{pattern.name}', generating mock")
        return self._generate_mock_result(pattern, search_region, active_states)

    def _get_snapshot_from_history(
        self, pattern: Pattern, active_states: set[str]
    ) -> ActionRecord | None:
        """Get a random snapshot from pattern's ActionHistory.

        Args:
            pattern: Pattern with ActionHistory
            active_states: Currently active states

        Returns:
            Random ActionRecord matching context, or None
        """
        if not hasattr(pattern, "match_history") or pattern.match_history.is_empty():
            return None

        # Get random snapshot preferring ones matching active states
        return cast(
            ActionRecord | None,
            pattern.match_history.get_random_snapshot(
                active_states=active_states, action_type="FIND"
            ),
        )

    def _create_result_from_snapshot(self, snapshot: ActionRecord) -> ActionResult:
        """Create ActionResult from historical snapshot.

        Args:
            snapshot: ActionRecord with historical match data

        Returns:
            ActionResult with matches from snapshot
        """
        builder = ActionResultBuilder()

        if snapshot.was_found():
            builder.with_success(True)
            # Set match list directly after building
            result = builder.build()
            result.set_match_list(snapshot.match_list)  # type: ignore[attr-defined,arg-type]

            if snapshot.text:
                result.add_text_result(snapshot.text)  # type: ignore[attr-defined]

            logger.debug(
                f"Created successful result with {len(snapshot.match_list)} matches"
            )
        else:
            builder.with_success(False)
            result = builder.build()
            logger.debug("Created failed result (snapshot had no matches)")

        duration_val = snapshot.duration if snapshot.duration > 0 else 0.1
        result.set_duration(timedelta(seconds=duration_val))  # type: ignore[attr-defined]
        return result

    def _get_result_from_api(
        self, pattern: Pattern, active_states: set[str]
    ) -> ActionResult | None:
        """Get historical result from qontinui-api.

        This method fetches a random historical result from the database
        for patterns that don't have local ActionHistory.

        Args:
            pattern: Pattern to find historical data for
            active_states: Currently active states

        Returns:
            ActionResult from API data, or None if not available
        """
        try:
            from ..mock.historical_data_client import get_historical_data_client

            client = get_historical_data_client()

            # Fetch random historical result matching pattern and context
            historical = client.get_random_historical_result(
                pattern_id=pattern.name,  # Use pattern name as ID
                action_type="FIND",
                active_states=list(active_states) if active_states else None,
                success_only=False,  # Include failures for realistic simulation
            )

            if not historical:
                return None

            if historical.success and historical.match_x is not None:
                # Create match from historical data
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
                result = ActionResultBuilder().with_success(True).build()
                result.add_match(match)  # type: ignore[attr-defined,arg-type]

                # Store the historical result ID for frame retrieval
                result.historical_result_id = historical.id  # type: ignore[attr-defined]

                logger.debug(
                    f"Created result from API: match at ({historical.match_x}, {historical.match_y})"
                )
            else:
                result = ActionResultBuilder().with_success(False).build()
                logger.debug("Created failed result from API historical data")

            return result

        except ImportError:
            logger.debug("Historical data client not available")
            return None
        except Exception as e:
            logger.warning(f"Error fetching historical data from API: {e}")
            return None

    def _generate_mock_result(
        self, pattern: Pattern, search_region: Region | None, active_states: set[str]
    ) -> ActionResult:
        """Generate mock result when no history exists.

        This fallback is used when patterns don't have ActionHistory yet.
        It uses state probabilities to determine success.

        Args:
            pattern: Pattern to mock
            search_region: Optional search region
            active_states: Currently active states

        Returns:
            Generated mock ActionResult
        """
        # Check if pattern's owner state is active
        if hasattr(pattern, "owner_state_name") and pattern.owner_state_name:
            if (
                pattern.owner_state_name not in active_states
                and pattern.owner_state_name != "NULL"
            ):
                # Pattern's state is not active
                result = ActionResultBuilder().with_success(False).build()
                logger.debug(f"Pattern's state '{pattern.owner_state_name}' not active")
                return result

        # Use state probability to determine if found
        probability = 1.0  # Default to always found
        if pattern.owner_state_name and pattern.owner_state_name != "NULL":
            probability = (
                self.state_management.get_state_probability(pattern.owner_state_name)
                / 100.0
            )

        if random.random() < probability:
            # Generate a mock match
            match = self._create_mock_match(pattern, search_region)
            result = ActionResultBuilder().with_success(True).build()
            result.add_match(match)  # type: ignore[attr-defined,arg-type]
            logger.debug(f"Generated mock match at {match.get_region()}")
        else:
            result = ActionResultBuilder().with_success(False).build()
            logger.debug(f"Mock find failed (probability was {probability})")

        return result

    def _create_mock_match(
        self, pattern: Pattern, search_region: Region | None
    ) -> Match:
        """Create a mock Match object.

        Args:
            pattern: Pattern that was "found"
            search_region: Region searched in

        Returns:
            Mock Match object
        """
        # Generate random position within search region or screen
        if search_region and search_region.is_defined():
            x = search_region.x + random.randint(0, max(0, search_region.width - 100))
            y = search_region.y + random.randint(0, max(0, search_region.height - 50))
        else:
            # Default screen area
            x = random.randint(100, 1820)
            y = random.randint(100, 980)

        # Create match with reasonable size
        width = random.randint(50, 200)
        height = random.randint(30, 100)

        match = Match(
            target=Location(region=Region(x, y, width, height)),
            score=random.uniform(0.85, 0.99),
            ocr_text="",
        )

        return match

    def find_all(
        self, pattern: Pattern, search_region: Region | None = None
    ) -> list[Match]:
        """Find all occurrences of a pattern.

        Args:
            pattern: Pattern to find
            search_region: Optional region to search in

        Returns:
            List of Match objects
        """
        result = self.find(pattern, search_region)
        # Access match_list directly to maintain correct type
        return result.match_list if result.success else []  # type: ignore[attr-defined,return-value]

    def wait_for(self, pattern: Pattern, timeout: float = 5.0) -> Match | None:
        """Simulate waiting for a pattern.

        Args:
            pattern: Pattern to wait for
            timeout: Maximum wait time

        Returns:
            Match if found, None if timeout
        """
        # In mock mode, decide immediately based on probability
        result = self.find(pattern)

        # Access match_list directly to maintain correct type
        if result.success and result.match_list:  # type: ignore[attr-defined]
            return result.match_list[0]  # type: ignore[attr-defined,return-value,no-any-return]

        return None
