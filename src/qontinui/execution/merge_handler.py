"""Merge handling for parallel execution paths.

This module handles merging of multiple execution paths that converge
at a single action node (merge point).
"""

import logging
from collections import defaultdict
from enum import Enum
from typing import Any

from ..config import Action, Connections

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """Strategy for merging execution paths."""

    WAIT_ALL = "wait_all"  # Wait for all incoming paths
    FIRST_WINS = "first_wins"  # Execute on first path arrival
    MAJORITY = "majority"  # Wait for majority of paths
    ANY_SUCCESS = "any_success"  # Execute if any path succeeded
    ALL_SUCCESS = "all_success"  # Execute only if all paths succeeded


class MergePoint:
    """Represents a merge point where multiple paths converge."""

    def __init__(
        self,
        action_id: str,
        incoming_paths: list[str],
        strategy: MergeStrategy = MergeStrategy.WAIT_ALL,
    ) -> None:
        """Initialize merge point.

        Args:
            action_id: ID of action that is the merge point
            incoming_paths: List of action IDs that lead to this merge point
            strategy: Merge strategy to use
        """
        self.action_id = action_id
        self.incoming_paths = incoming_paths
        self.strategy = strategy
        self.arrived_paths: set[str] = set()
        self.path_results: dict[str, dict[str, Any]] = {}
        self.is_ready = False
        self.merged_context: dict[str, Any] | None = None

        logger.debug(
            f"Created merge point at '{action_id}' with {len(incoming_paths)} incoming paths"
        )

    def register_arrival(self, from_action_id: str, result: dict[str, Any]) -> bool:
        """Register arrival of an execution path.

        Args:
            from_action_id: ID of action that completed
            result: Execution result from that action

        Returns:
            True if merge point is now ready to execute
        """
        if from_action_id not in self.incoming_paths:
            logger.warning(
                f"Unexpected arrival at merge point '{self.action_id}' from '{from_action_id}'"
            )
            return False

        self.arrived_paths.add(from_action_id)
        self.path_results[from_action_id] = result

        logger.debug(
            f"Merge point '{self.action_id}': {len(self.arrived_paths)}/{len(self.incoming_paths)} paths arrived"
        )

        # Check if ready based on strategy
        self.is_ready = self._check_ready()
        return self.is_ready

    def _check_ready(self) -> bool:
        """Check if merge point is ready based on strategy."""
        if self.strategy == MergeStrategy.WAIT_ALL:
            return len(self.arrived_paths) == len(self.incoming_paths)

        elif self.strategy == MergeStrategy.FIRST_WINS:
            return len(self.arrived_paths) >= 1

        elif self.strategy == MergeStrategy.MAJORITY:
            return len(self.arrived_paths) > len(self.incoming_paths) / 2

        elif self.strategy == MergeStrategy.ANY_SUCCESS:
            # Ready if any path succeeded
            return any(result.get("success", False) for result in self.path_results.values())

        elif self.strategy == MergeStrategy.ALL_SUCCESS:
            # Ready if all arrived paths succeeded AND all paths have arrived
            if len(self.arrived_paths) < len(self.incoming_paths):
                return False
            return all(result.get("success", False) for result in self.path_results.values())

        return False

    def get_merged_context(self) -> dict[str, Any]:
        """Get merged context from all arrived paths.

        Returns:
            Merged context dictionary
        """
        if self.merged_context is not None:
            return self.merged_context

        # Start with empty context
        merged = {}

        # Merge contexts from all paths
        for _from_action, result in self.path_results.items():
            context = result.get("context", {})

            # Simple merge: later paths override earlier ones
            # In practice, you might want more sophisticated merging
            for key, value in context.items():
                if key in merged and merged[key] != value:
                    logger.debug(
                        f"Context conflict at merge point '{self.action_id}': "
                        f"key='{key}', existing={merged[key]}, new={value}"
                    )
                merged[key] = value

        self.merged_context = merged
        return merged

    def get_merge_summary(self) -> dict[str, Any]:
        """Get summary of merge state.

        Returns:
            Dictionary with merge information
        """
        return {
            "action_id": self.action_id,
            "strategy": self.strategy,
            "total_paths": len(self.incoming_paths),
            "arrived_paths": len(self.arrived_paths),
            "is_ready": self.is_ready,
            "path_results": {
                action_id: {"success": result.get("success", False)}
                for action_id, result in self.path_results.items()
            },
        }


class MergeHandler:
    """Manages merge points in workflow graph.

    The MergeHandler is responsible for:
    - Detecting merge points (actions with multiple incoming connections)
    - Tracking execution path arrivals at merge points
    - Determining when merged actions are ready to execute
    - Merging contexts from multiple paths

    Attributes:
        connections: Connection graph from workflow
        action_map: Quick lookup for actions by ID
        merge_points: Dictionary of merge point objects
    """

    def __init__(self, connections: Connections, action_map: dict[str, Action]) -> None:
        """Initialize merge handler.

        Args:
            connections: Connection graph from workflow
            action_map: Dictionary mapping action ID to Action
        """
        self.connections = connections
        self.action_map = action_map
        self.merge_points: dict[str, MergePoint] = {}

        # Detect and initialize merge points
        self._detect_merge_points()

        logger.info(f"Initialized MergeHandler with {len(self.merge_points)} merge points")

    def _detect_merge_points(self):
        """Detect all merge points in the workflow.

        A merge point is an action that has multiple incoming connections
        from different source actions.
        """
        # Count incoming connections for each action
        incoming_count: dict[str, list[str]] = defaultdict(list)

        for source_id, conn_types in self.connections.root.items():
            for _conn_type, conn_lists in conn_types.items():
                for conn_list in conn_lists:
                    for connection in conn_list:
                        target_id = connection.action
                        if source_id not in incoming_count[target_id]:
                            incoming_count[target_id].append(source_id)

        # Create merge points for actions with multiple incoming paths
        for action_id, source_actions in incoming_count.items():
            if len(source_actions) > 1:
                merge_point = MergePoint(
                    action_id=action_id,
                    incoming_paths=source_actions,
                    strategy=self._determine_merge_strategy(action_id),
                )
                self.merge_points[action_id] = merge_point

                logger.debug(
                    f"Detected merge point: '{action_id}' "
                    f"(incoming from {len(source_actions)} actions)"
                )

    def _determine_merge_strategy(self, action_id: str) -> MergeStrategy:
        """Determine appropriate merge strategy for an action.

        This can be customized based on action type, configuration, or
        workflow settings. Default is WAIT_ALL.

        Args:
            action_id: Action ID to determine strategy for

        Returns:
            MergeStrategy to use
        """
        # For now, always use WAIT_ALL
        # In the future, this could be based on:
        # - Action metadata
        # - Workflow settings
        # - Connection types
        return MergeStrategy.WAIT_ALL

    def is_merge_point(self, action_id: str) -> bool:
        """Check if an action is a merge point.

        Args:
            action_id: Action ID to check

        Returns:
            True if action is a merge point
        """
        return action_id in self.merge_points

    def register_arrival(
        self, merge_action_id: str, from_action_id: str, result: dict[str, Any]
    ) -> bool:
        """Register arrival of execution path at merge point.

        Args:
            merge_action_id: ID of merge point action
            from_action_id: ID of action that just completed
            result: Execution result from completed action

        Returns:
            True if merge point is now ready to execute
        """
        if not self.is_merge_point(merge_action_id):
            logger.warning(f"Attempted to register arrival at non-merge-point '{merge_action_id}'")
            return True  # Not a merge point, so it's always "ready"

        merge_point = self.merge_points[merge_action_id]
        is_ready = merge_point.register_arrival(from_action_id, result)

        if is_ready:
            logger.info(
                f"Merge point '{merge_action_id}' is ready to execute "
                f"({len(merge_point.arrived_paths)}/{len(merge_point.incoming_paths)} paths)"
            )

        return is_ready

    def get_merged_context(self, merge_action_id: str) -> dict[str, Any]:
        """Get merged context for a merge point.

        Args:
            merge_action_id: ID of merge point action

        Returns:
            Merged context dictionary
        """
        if not self.is_merge_point(merge_action_id):
            return {}

        merge_point = self.merge_points[merge_action_id]
        return merge_point.get_merged_context()

    def reset_merge_point(self, merge_action_id: str):
        """Reset a merge point (clear arrived paths).

        This is useful for loops or repeated executions.

        Args:
            merge_action_id: ID of merge point to reset
        """
        if not self.is_merge_point(merge_action_id):
            return

        merge_point = self.merge_points[merge_action_id]
        merge_point.arrived_paths.clear()
        merge_point.path_results.clear()
        merge_point.is_ready = False
        merge_point.merged_context = None

        logger.debug(f"Reset merge point '{merge_action_id}'")

    def get_merge_status(self, merge_action_id: str) -> dict[str, Any] | None:
        """Get current status of a merge point.

        Args:
            merge_action_id: ID of merge point

        Returns:
            Dictionary with merge status, or None if not a merge point
        """
        if not self.is_merge_point(merge_action_id):
            return None

        merge_point = self.merge_points[merge_action_id]
        return merge_point.get_merge_summary()

    def get_all_merge_points(self) -> list[str]:
        """Get list of all merge point action IDs.

        Returns:
            List of action IDs that are merge points
        """
        return list(self.merge_points.keys())

    def get_blocking_paths(self, merge_action_id: str) -> list[str]:
        """Get list of paths that haven't arrived yet.

        Args:
            merge_action_id: ID of merge point

        Returns:
            List of action IDs that haven't completed yet
        """
        if not self.is_merge_point(merge_action_id):
            return []

        merge_point = self.merge_points[merge_action_id]
        return [
            path for path in merge_point.incoming_paths if path not in merge_point.arrived_paths
        ]

    def validate_merge_points(self) -> tuple[bool, list[str]]:
        """Validate all merge points.

        Checks for:
        - Unreachable merge points
        - Circular dependencies in merge points
        - Invalid incoming connections

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []

        for action_id, merge_point in self.merge_points.items():
            # Check all incoming paths exist
            for incoming_id in merge_point.incoming_paths:
                if incoming_id not in self.action_map:
                    warnings.append(
                        f"Merge point '{action_id}' has incoming path from "
                        f"non-existent action '{incoming_id}'"
                    )

            # Check merge point action exists
            if action_id not in self.action_map:
                warnings.append(f"Merge point action '{action_id}' does not exist")

        return len(warnings) == 0, warnings

    def get_merge_statistics(self) -> dict[str, Any]:
        """Get statistics about merge points.

        Returns:
            Dictionary with merge statistics
        """
        if not self.merge_points:
            return {"total_merge_points": 0, "ready": 0, "waiting": 0}

        ready_count = sum(1 for mp in self.merge_points.values() if mp.is_ready)

        return {
            "total_merge_points": len(self.merge_points),
            "ready": ready_count,
            "waiting": len(self.merge_points) - ready_count,
            "merge_points": [
                {
                    "action_id": action_id,
                    "incoming_count": len(mp.incoming_paths),
                    "arrived_count": len(mp.arrived_paths),
                    "is_ready": mp.is_ready,
                }
                for action_id, mp in self.merge_points.items()
            ],
        }
