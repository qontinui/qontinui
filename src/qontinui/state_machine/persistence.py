"""State Persistence for UI Bridge Integration.

This module provides persistence for discovered states and transitions,
using the runner's existing database tables:
- ui_bridge_states
- ui_bridge_transitions
- ui_bridge_state_groups

The persistence layer allows states/transitions to survive application
restarts and be shared across different sessions.

Example:
    from qontinui.state_machine import StatePersistence, UIBridgeState

    # Initialize with database path
    persistence = StatePersistence("path/to/qontinui-runner.db")

    # Save discovered states
    state = UIBridgeState(
        id="dashboard_state",
        name="Dashboard",
        element_ids=["nav-menu", "sidebar", "main-content"],
    )
    persistence.save_state(state)

    # Load states for runtime
    states = persistence.load_states()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .ui_bridge_runtime import UIBridgeState, UIBridgeTransition

logger = logging.getLogger(__name__)


@dataclass
class StateGroupRecord:
    """Represents a state group from the database."""

    id: str
    name: str
    state_ids: list[str]
    metadata: dict[str, Any]


class StatePersistence:
    """Persistence layer for UI Bridge states and transitions.

    Uses the runner's SQLite database with existing schema:
    - ui_bridge_states: State definitions
    - ui_bridge_transitions: Transition definitions
    - ui_bridge_state_groups: State groups

    Thread-safe via connection pooling.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize persistence layer.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_tables()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection.

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_tables(self) -> None:
        """Ensure required tables exist.

        The tables should already exist from the runner schema,
        but this provides a fallback for standalone usage.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ui_bridge_states'"
            )
            if cursor.fetchone() is None:
                logger.warning(
                    "UI Bridge tables not found in database. "
                    "This database may not be a qontinui-runner database."
                )

    # =========================================================================
    # State Operations
    # =========================================================================

    def save_state(self, state: UIBridgeState) -> None:
        """Save or update a state.

        Args:
            state: State to save
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            elements_json = json.dumps(state.element_ids)
            blocks_json = json.dumps(state.blocks)
            metadata_json = json.dumps(state.metadata)

            cursor.execute(
                """
                INSERT INTO ui_bridge_states (
                    state_id, name, elements, blocking, blocks, group_id,
                    path_cost, metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(state_id) DO UPDATE SET
                    name = excluded.name,
                    elements = excluded.elements,
                    blocking = excluded.blocking,
                    blocks = excluded.blocks,
                    group_id = excluded.group_id,
                    path_cost = excluded.path_cost,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    state.id,
                    state.name,
                    elements_json,
                    1 if state.blocking else 0,
                    blocks_json,
                    state.group,
                    state.path_cost,
                    metadata_json,
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            logger.debug(f"Saved state: {state.id}")

    def save_states(self, states: list[UIBridgeState]) -> None:
        """Save multiple states in a batch.

        Args:
            states: States to save
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for state in states:
                elements_json = json.dumps(state.element_ids)
                blocks_json = json.dumps(state.blocks)
                metadata_json = json.dumps(state.metadata)

                cursor.execute(
                    """
                    INSERT INTO ui_bridge_states (
                        state_id, name, elements, blocking, blocks, group_id,
                        path_cost, metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(state_id) DO UPDATE SET
                        name = excluded.name,
                        elements = excluded.elements,
                        blocking = excluded.blocking,
                        blocks = excluded.blocks,
                        group_id = excluded.group_id,
                        path_cost = excluded.path_cost,
                        metadata = excluded.metadata,
                        updated_at = excluded.updated_at
                    """,
                    (
                        state.id,
                        state.name,
                        elements_json,
                        1 if state.blocking else 0,
                        blocks_json,
                        state.group,
                        state.path_cost,
                        metadata_json,
                        datetime.utcnow().isoformat(),
                    ),
                )

            conn.commit()
            logger.info(f"Saved {len(states)} states")

    def load_state(self, state_id: str) -> UIBridgeState | None:
        """Load a state by ID.

        Args:
            state_id: State ID

        Returns:
            UIBridgeState or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT state_id, name, elements, blocking, blocks, group_id,
                       path_cost, metadata
                FROM ui_bridge_states
                WHERE state_id = ?
                """,
                (state_id,),
            )

            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_state(row)

    def load_states(self) -> list[UIBridgeState]:
        """Load all states.

        Returns:
            List of all stored states
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT state_id, name, elements, blocking, blocks, group_id,
                       path_cost, metadata
                FROM ui_bridge_states
                ORDER BY name
                """
            )

            return [self._row_to_state(row) for row in cursor.fetchall()]

    def load_states_in_group(self, group_id: str) -> list[UIBridgeState]:
        """Load all states in a group.

        Args:
            group_id: Group ID

        Returns:
            List of states in the group
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT state_id, name, elements, blocking, blocks, group_id,
                       path_cost, metadata
                FROM ui_bridge_states
                WHERE group_id = ?
                ORDER BY name
                """,
                (group_id,),
            )

            return [self._row_to_state(row) for row in cursor.fetchall()]

    def delete_state(self, state_id: str) -> bool:
        """Delete a state.

        Args:
            state_id: State ID to delete

        Returns:
            True if state was deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM ui_bridge_states WHERE state_id = ?",
                (state_id,),
            )

            conn.commit()
            deleted = cursor.rowcount > 0

            if deleted:
                logger.debug(f"Deleted state: {state_id}")

            return deleted

    def _row_to_state(self, row: sqlite3.Row) -> UIBridgeState:
        """Convert database row to UIBridgeState."""
        return UIBridgeState(
            id=row["state_id"],
            name=row["name"],
            element_ids=json.loads(row["elements"] or "[]"),
            blocking=bool(row["blocking"]),
            blocks=json.loads(row["blocks"] or "[]"),
            group=row["group_id"],
            path_cost=row["path_cost"] or 1.0,
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # =========================================================================
    # Transition Operations
    # =========================================================================

    def save_transition(self, transition: UIBridgeTransition) -> None:
        """Save or update a transition.

        Args:
            transition: Transition to save
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            from_states_json = json.dumps(transition.from_states)
            activate_states_json = json.dumps(transition.activate_states)
            exit_states_json = json.dumps(transition.exit_states)
            actions_json = json.dumps(transition.actions)
            activate_groups_json = json.dumps(transition.activate_groups)
            exit_groups_json = json.dumps(transition.exit_groups)
            metadata_json = json.dumps(transition.metadata)

            cursor.execute(
                """
                INSERT INTO ui_bridge_transitions (
                    transition_id, name, from_states, activate_states, exit_states,
                    actions, activate_groups, exit_groups, path_cost, stays_visible,
                    metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transition_id) DO UPDATE SET
                    name = excluded.name,
                    from_states = excluded.from_states,
                    activate_states = excluded.activate_states,
                    exit_states = excluded.exit_states,
                    actions = excluded.actions,
                    activate_groups = excluded.activate_groups,
                    exit_groups = excluded.exit_groups,
                    path_cost = excluded.path_cost,
                    stays_visible = excluded.stays_visible,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    transition.id,
                    transition.name,
                    from_states_json,
                    activate_states_json,
                    exit_states_json,
                    actions_json,
                    activate_groups_json,
                    exit_groups_json,
                    transition.path_cost,
                    1 if transition.stays_visible else 0,
                    metadata_json,
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            logger.debug(f"Saved transition: {transition.id}")

    def save_transitions(self, transitions: list[UIBridgeTransition]) -> None:
        """Save multiple transitions in a batch.

        Args:
            transitions: Transitions to save
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for transition in transitions:
                from_states_json = json.dumps(transition.from_states)
                activate_states_json = json.dumps(transition.activate_states)
                exit_states_json = json.dumps(transition.exit_states)
                actions_json = json.dumps(transition.actions)
                activate_groups_json = json.dumps(transition.activate_groups)
                exit_groups_json = json.dumps(transition.exit_groups)
                metadata_json = json.dumps(transition.metadata)

                cursor.execute(
                    """
                    INSERT INTO ui_bridge_transitions (
                        transition_id, name, from_states, activate_states, exit_states,
                        actions, activate_groups, exit_groups, path_cost, stays_visible,
                        metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(transition_id) DO UPDATE SET
                        name = excluded.name,
                        from_states = excluded.from_states,
                        activate_states = excluded.activate_states,
                        exit_states = excluded.exit_states,
                        actions = excluded.actions,
                        activate_groups = excluded.activate_groups,
                        exit_groups = excluded.exit_groups,
                        path_cost = excluded.path_cost,
                        stays_visible = excluded.stays_visible,
                        metadata = excluded.metadata,
                        updated_at = excluded.updated_at
                    """,
                    (
                        transition.id,
                        transition.name,
                        from_states_json,
                        activate_states_json,
                        exit_states_json,
                        actions_json,
                        activate_groups_json,
                        exit_groups_json,
                        transition.path_cost,
                        1 if transition.stays_visible else 0,
                        metadata_json,
                        datetime.utcnow().isoformat(),
                    ),
                )

            conn.commit()
            logger.info(f"Saved {len(transitions)} transitions")

    def load_transition(self, transition_id: str) -> UIBridgeTransition | None:
        """Load a transition by ID.

        Args:
            transition_id: Transition ID

        Returns:
            UIBridgeTransition or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT transition_id, name, from_states, activate_states, exit_states,
                       actions, activate_groups, exit_groups, path_cost, stays_visible,
                       metadata
                FROM ui_bridge_transitions
                WHERE transition_id = ?
                """,
                (transition_id,),
            )

            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_transition(row)

    def load_transitions(self) -> list[UIBridgeTransition]:
        """Load all transitions.

        Returns:
            List of all stored transitions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT transition_id, name, from_states, activate_states, exit_states,
                       actions, activate_groups, exit_groups, path_cost, stays_visible,
                       metadata
                FROM ui_bridge_transitions
                ORDER BY name
                """
            )

            return [self._row_to_transition(row) for row in cursor.fetchall()]

    def load_transitions_from_state(self, state_id: str) -> list[UIBridgeTransition]:
        """Load transitions that can execute from a state.

        Args:
            state_id: State ID

        Returns:
            List of available transitions
        """
        transitions = self.load_transitions()
        return [t for t in transitions if state_id in t.from_states]

    def delete_transition(self, transition_id: str) -> bool:
        """Delete a transition.

        Args:
            transition_id: Transition ID to delete

        Returns:
            True if transition was deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM ui_bridge_transitions WHERE transition_id = ?",
                (transition_id,),
            )

            conn.commit()
            deleted = cursor.rowcount > 0

            if deleted:
                logger.debug(f"Deleted transition: {transition_id}")

            return deleted

    def _row_to_transition(self, row: sqlite3.Row) -> UIBridgeTransition:
        """Convert database row to UIBridgeTransition."""
        return UIBridgeTransition(
            id=row["transition_id"],
            name=row["name"],
            from_states=json.loads(row["from_states"] or "[]"),
            activate_states=json.loads(row["activate_states"] or "[]"),
            exit_states=json.loads(row["exit_states"] or "[]"),
            actions=json.loads(row["actions"] or "[]"),
            activate_groups=json.loads(row["activate_groups"] or "[]"),
            exit_groups=json.loads(row["exit_groups"] or "[]"),
            path_cost=row["path_cost"] or 1.0,
            stays_visible=bool(row["stays_visible"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # =========================================================================
    # State Group Operations
    # =========================================================================

    def save_state_group(
        self, group_id: str, name: str, state_ids: list[str], metadata: dict[str, Any] | None = None
    ) -> None:
        """Save or update a state group.

        Args:
            group_id: Group ID
            name: Group name
            state_ids: State IDs in the group
            metadata: Optional metadata
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            states_json = json.dumps(state_ids)
            metadata_json = json.dumps(metadata or {})

            cursor.execute(
                """
                INSERT INTO ui_bridge_state_groups (
                    group_id, name, states, metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    name = excluded.name,
                    states = excluded.states,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    group_id,
                    name,
                    states_json,
                    metadata_json,
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            logger.debug(f"Saved state group: {group_id}")

    def load_state_groups(self) -> list[StateGroupRecord]:
        """Load all state groups.

        Returns:
            List of all stored state groups
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT group_id, name, states, metadata
                FROM ui_bridge_state_groups
                ORDER BY name
                """
            )

            groups = []
            for row in cursor.fetchall():
                groups.append(
                    StateGroupRecord(
                        id=row["group_id"],
                        name=row["name"],
                        state_ids=json.loads(row["states"] or "[]"),
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )

            return groups

    def delete_state_group(self, group_id: str) -> bool:
        """Delete a state group.

        Args:
            group_id: Group ID to delete

        Returns:
            True if group was deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM ui_bridge_state_groups WHERE group_id = ?",
                (group_id,),
            )

            conn.commit()
            deleted = cursor.rowcount > 0

            if deleted:
                logger.debug(f"Deleted state group: {group_id}")

            return deleted

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def clear_all(self) -> None:
        """Clear all states, transitions, and groups."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM ui_bridge_states")
            cursor.execute("DELETE FROM ui_bridge_transitions")
            cursor.execute("DELETE FROM ui_bridge_state_groups")

            conn.commit()
            logger.info("Cleared all UI Bridge state machine data")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about persisted data.

        Returns:
            Dictionary with counts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM ui_bridge_states")
            state_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM ui_bridge_transitions")
            transition_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM ui_bridge_state_groups")
            group_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(DISTINCT group_id) FROM ui_bridge_states WHERE group_id IS NOT NULL"
            )
            states_with_groups = cursor.fetchone()[0]

            return {
                "states": state_count,
                "transitions": transition_count,
                "groups": group_count,
                "states_with_groups": states_with_groups,
            }
