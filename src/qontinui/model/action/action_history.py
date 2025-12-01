"""ActionHistory class - ported from Brobot framework.

Manages the complete history of actions for analysis and mocking.
"""

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .action_record import ActionRecord


@dataclass
class ActionHistory:
    """Manages the complete history of actions for analysis and mocking.

    Port of ActionHistory from Qontinui framework class (simplified).

    ActionHistory is the central repository for all ActionRecords generated during
    automation execution. It provides comprehensive analysis capabilities, enabling
    the framework to learn from past executions, optimize configurations, and provide
    realistic mock responses during development and testing.

    Key capabilities:
    - Historical Storage: Maintains chronological record of all actions
    - Statistical Analysis: Computes success rates, timing distributions
    - Pattern Recognition: Identifies trends in match behavior
    - Mock Support: Provides data for realistic simulation
    - Performance Metrics: Tracks execution times and efficiency

    Analysis dimensions:
    - By State: Success rates and patterns within specific states
    - By Action Type: Performance metrics for different actions
    - By Time: Temporal patterns and degradation analysis
    - By Configuration: Impact of different settings on success

    Mock operation support:
    - Realistic failure simulation based on historical rates
    - Accurate match count distributions
    - Representative timing variations
    - State-specific behavior patterns
    """

    records: list[ActionRecord] = field(default_factory=list)
    _records_by_state: dict[str, list[ActionRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _records_by_action: dict[str, list[ActionRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    max_records: int = 10000  # Maximum records to keep

    def add_record(self, record: ActionRecord) -> None:
        """Add a new action record to history.

        Args:
            record: ActionRecord to add
        """
        self.records.append(record)

        # Index by state
        self._records_by_state[record.state_name].append(record)

        # Index by action type
        action_type = record.get_action_type()
        self._records_by_action[action_type].append(record)

        # Limit history size
        if len(self.records) > self.max_records:
            removed = self.records.pop(0)
            # Also remove from indices
            self._records_by_state[removed.state_name].remove(removed)
            self._records_by_action[removed.get_action_type()].remove(removed)

    def get_records_by_state(self, state_name: str) -> list[ActionRecord]:
        """Get all records for a specific state.

        Args:
            state_name: Name of state

        Returns:
            List of ActionRecords for that state
        """
        return self._records_by_state.get(state_name, [])

    def get_records_by_action(self, action_type: str) -> list[ActionRecord]:
        """Get all records for a specific action type.

        Args:
            action_type: Type of action (CLICK, FIND, etc.)

        Returns:
            List of ActionRecords for that action
        """
        return self._records_by_action.get(action_type, [])

    def get_recent_records(self, count: int = 10) -> list[ActionRecord]:
        """Get most recent records.

        Args:
            count: Number of records to return

        Returns:
            List of most recent ActionRecords
        """
        return self.records[-count:] if self.records else []

    def get_success_rate(
        self, state_name: str | None = None, action_type: str | None = None
    ) -> float:
        """Calculate success rate for actions.

        Args:
            state_name: Filter by state (optional)
            action_type: Filter by action type (optional)

        Returns:
            Success rate as percentage (0.0-1.0)
        """
        records = self._filter_records(state_name, action_type)
        if not records:
            return 0.0

        successful = sum(1 for r in records if r.action_success)
        return successful / len(records)

    def get_average_duration(
        self, state_name: str | None = None, action_type: str | None = None
    ) -> float:
        """Calculate average action duration.

        Args:
            state_name: Filter by state (optional)
            action_type: Filter by action type (optional)

        Returns:
            Average duration in seconds
        """
        records = self._filter_records(state_name, action_type)
        if not records:
            return 0.0

        durations = [r.duration for r in records if r.duration > 0]
        return statistics.mean(durations) if durations else 0.0

    def get_match_count_distribution(
        self, state_name: str | None = None, action_type: str | None = None
    ) -> dict[int, int]:
        """Get distribution of match counts.

        Args:
            state_name: Filter by state (optional)
            action_type: Filter by action type (optional)

        Returns:
            Dict mapping match count to frequency
        """
        records = self._filter_records(state_name, action_type)
        distribution: defaultdict[int, int] = defaultdict(int)

        for record in records:
            match_count = len(record.match_list)
            distribution[match_count] += 1

        return dict(distribution)

    def get_failure_patterns(self, window_minutes: int = 60) -> list[dict[str, Any]]:
        """Analyze recent failure patterns.

        Args:
            window_minutes: Time window to analyze

        Returns:
            List of failure pattern information
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_records = [r for r in self.records if r.timestamp > cutoff_time]

        patterns = []
        consecutive_failures = 0
        failure_start = None

        for record in recent_records:
            if not record.action_success:
                if consecutive_failures == 0:
                    failure_start = record
                consecutive_failures += 1
            else:
                if consecutive_failures >= 3 and failure_start is not None:  # Pattern threshold
                    patterns.append(
                        {
                            "start": failure_start.timestamp,
                            "count": consecutive_failures,
                            "state": failure_start.state_name,
                            "action": failure_start.get_action_type(),
                        }
                    )
                consecutive_failures = 0
                failure_start = None

        return patterns

    def get_mock_record(self, state_name: str, action_type: str) -> ActionRecord | None:
        """Get a representative record for mocking.

        Args:
            state_name: State to mock
            action_type: Action type to mock

        Returns:
            Representative ActionRecord or None
        """
        records = self._filter_records(state_name, action_type)
        if not records:
            return None

        # Return most recent successful record, or most recent if none successful
        successful = [r for r in records if r.action_success]
        if successful:
            return successful[-1]
        return records[-1]

    def get_random_snapshot(
        self, active_states: set[str] | None = None, action_type: str = "FIND"
    ) -> ActionRecord | None:
        """Get a random snapshot for mocking, filtered by active states.

        This is the key method for Brobot-style mocking. It returns a random
        ActionRecord from the history, preferring records that match the
        current active states.

        Args:
            active_states: Set of currently active state names
            action_type: Type of action to filter by

        Returns:
            Random ActionRecord or None if no matches
        """
        if not self.records:
            return None

        # Filter by action type
        records = [r for r in self.records if r.get_action_type() == action_type]
        if not records:
            return None

        # If states provided, prefer records matching those states
        if active_states:
            # First try exact state matches
            state_matches = [r for r in records if r.state_name in active_states]
            if state_matches:
                return random.choice(state_matches)

        # Fall back to any record of the right type
        return random.choice(records)

    def add_snapshot(self, snapshot: ActionRecord) -> None:
        """Add a snapshot (alias for add_record for Brobot compatibility).

        Args:
            snapshot: ActionRecord to add
        """
        self.add_record(snapshot)

    def get_snapshots(self) -> list[ActionRecord]:
        """Get all snapshots (alias for records for Brobot compatibility).

        Returns:
            List of all ActionRecords
        """
        return self.records

    def is_empty(self) -> bool:
        """Check if history is empty.

        Returns:
            True if no records exist
        """
        return len(self.records) == 0

    def merge(self, other: "ActionHistory") -> None:
        """Merge another ActionHistory into this one.

        Args:
            other: ActionHistory to merge in
        """
        for record in other.records:
            self.add_record(record)

    def _filter_records(
        self, state_name: str | None = None, action_type: str | None = None
    ) -> list[ActionRecord]:
        """Filter records by criteria.

        Args:
            state_name: Filter by state (optional)
            action_type: Filter by action type (optional)

        Returns:
            Filtered list of ActionRecords
        """
        records = self.records

        if state_name:
            records = [r for r in records if r.state_name == state_name]

        if action_type:
            records = [r for r in records if r.get_action_type() == action_type]

        return records

    def clear(self) -> None:
        """Clear all history."""
        self.records.clear()
        self._records_by_state.clear()
        self._records_by_action.clear()

    def size(self) -> int:
        """Get number of records.

        Returns:
            Total number of ActionRecords
        """
        return len(self.records)

    def get_states(self) -> set[str]:
        """Get all states in history.

        Returns:
            Set of state names
        """
        return set(self._records_by_state.keys())

    def get_action_types(self) -> set[str]:
        """Get all action types in history.

        Returns:
            Set of action type names
        """
        return set(self._records_by_action.keys())

    def print_summary(self) -> None:
        """Print summary statistics."""
        print("ActionHistory Summary:")
        print(f"  Total records: {self.size()}")
        print(f"  States: {', '.join(self.get_states())}")
        print(f"  Action types: {', '.join(self.get_action_types())}")
        print(f"  Overall success rate: {self.get_success_rate():.1%}")
        print(f"  Average duration: {self.get_average_duration():.3f}s")

        # Per-state statistics
        for state in self.get_states():
            records = self.get_records_by_state(state)
            if records:
                print(f"\n  {state}:")
                print(f"    Records: {len(records)}")
                print(f"    Success rate: {self.get_success_rate(state_name=state):.1%}")
                print(f"    Avg duration: {self.get_average_duration(state_name=state):.3f}s")

    def __str__(self) -> str:
        """String representation."""
        return f"ActionHistory({self.size()} records, {len(self.get_states())} states)"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"ActionHistory(records={self.size()}, "
            f"states={self.get_states()}, "
            f"actions={self.get_action_types()})"
        )
