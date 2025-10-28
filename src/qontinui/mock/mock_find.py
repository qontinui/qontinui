"""MockFind - Provides realistic pattern finding simulation using ActionHistory.

This module implements Brobot-style mocking where ActionHistory (containing ActionRecords)
drives the mock behavior, not manual probability settings.
"""

import logging
import random
from datetime import timedelta
from typing import cast

from ..actions.action_result import ActionResult
from ..mock.mock_mode_manager import MockModeManager
from ..mock.mock_state_management import MockStateManagement
from ..model.action import ActionRecord
from ..model.element import Location, Pattern, Region
from ..model.match import Match

logger = logging.getLogger(__name__)


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

    def find(self, pattern: Pattern, search_region: Region | None = None) -> ActionResult:
        """Simulate finding a pattern using its ActionHistory.

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

        # Try to get a snapshot from ActionHistory
        snapshot = self._get_snapshot_from_history(pattern, active_states)

        if snapshot:
            # Use historical data
            logger.debug(f"Using historical snapshot with {len(snapshot.match_list)} matches")
            return self._create_result_from_snapshot(snapshot)

        # No history - generate mock based on state probability
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
        result = ActionResult()

        if snapshot.was_found():
            result.success = True
            result.set_match_list(snapshot.match_list)  # type: ignore[arg-type]

            if snapshot.text:
                result.text = snapshot.text  # type: ignore[assignment]

            logger.debug(f"Created successful result with {len(snapshot.match_list)} matches")
        else:
            result.success = False
            logger.debug("Created failed result (snapshot had no matches)")

        duration_val = snapshot.duration if snapshot.duration > 0 else 0.1
        result.duration = timedelta(seconds=duration_val)
        return result

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
        result = ActionResult()

        # Check if pattern's owner state is active
        if hasattr(pattern, "owner_state_name") and pattern.owner_state_name:
            if pattern.owner_state_name not in active_states and pattern.owner_state_name != "NULL":
                # Pattern's state is not active
                result.success = False
                logger.debug(f"Pattern's state '{pattern.owner_state_name}' not active")
                return result

        # Use state probability to determine if found
        probability = 1.0  # Default to always found
        if pattern.owner_state_name and pattern.owner_state_name != "NULL":
            probability = (
                self.state_management.get_state_probability(pattern.owner_state_name) / 100.0
            )

        if random.random() < probability:
            # Generate a mock match
            match = self._create_mock_match(pattern, search_region)
            result.success = True
            result.add_match(match)  # type: ignore[arg-type]
            logger.debug(f"Generated mock match at {match.get_region()}")
        else:
            result.success = False
            logger.debug(f"Mock find failed (probability was {probability})")

        return result

    def _create_mock_match(self, pattern: Pattern, search_region: Region | None) -> Match:
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

    def find_all(self, pattern: Pattern, search_region: Region | None = None) -> list[Match]:
        """Find all occurrences of a pattern.

        Args:
            pattern: Pattern to find
            search_region: Optional region to search in

        Returns:
            List of Match objects
        """
        result = self.find(pattern, search_region)
        # Access match_list directly to maintain correct type
        return result.match_list if result.success else []  # type: ignore[return-value]

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
        if result.success and result.match_list:
            return result.match_list[0]  # type: ignore[return-value]

        return None
