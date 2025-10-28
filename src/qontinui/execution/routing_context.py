"""
Routing context for tracking execution flow through workflow graph.

Records routing decisions, execution paths, and provides debugging/visualization support.

Clean design for understanding and debugging workflow execution.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RouteRecord:
    """Record of a single routing decision."""

    from_action: str
    to_action: str
    output_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    input_index: int = 0
    execution_result: dict | None = None

    def __str__(self) -> str:
        """String representation of route record."""
        return f"{self.from_action} --[{self.output_type}]--> {self.to_action}"


@dataclass
class PathSegment:
    """Segment of an execution path."""

    action_id: str
    entry_output: str = "main"
    exit_output: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class RoutingContext:
    """
    Tracks routing decisions during workflow execution.

    Maintains execution history, path information, and provides
    analysis methods for debugging and visualization.
    """

    def __init__(self) -> None:
        """Initialize routing context."""
        self.records: list[RouteRecord] = []
        self.current_path: list[PathSegment] = []
        self.visited_actions: set[str] = set()
        self.action_visit_count: dict[str, int] = defaultdict(int)
        self.output_usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.start_time: datetime = datetime.now()
        self.end_time: datetime | None = None

    def record_route(
        self,
        from_action: str,
        to_action: str,
        output_type: str,
        input_index: int = 0,
        execution_result: dict | None = None,
    ) -> None:
        """
        Record a routing decision.

        Args:
            from_action: Source action ID
            to_action: Target action ID
            output_type: Connection type used (e.g., 'main', 'true', 'error')
            input_index: Input index on target action
            execution_result: Optional execution result that led to this route
        """
        record = RouteRecord(
            from_action=from_action,
            to_action=to_action,
            output_type=output_type,
            input_index=input_index,
            execution_result=execution_result,
        )

        self.records.append(record)
        self.visited_actions.add(from_action)
        self.visited_actions.add(to_action)
        self.action_visit_count[from_action] += 1
        self.output_usage[from_action][output_type] += 1

    def enter_action(self, action_id: str, output_type: str = "main") -> None:
        """
        Record entering an action.

        Args:
            action_id: Action being entered
            output_type: How we entered (which output led here)
        """
        segment = PathSegment(action_id=action_id, entry_output=output_type)
        self.current_path.append(segment)
        self.visited_actions.add(action_id)
        self.action_visit_count[action_id] += 1

    def exit_action(self, action_id: str, output_type: str) -> None:
        """
        Record exiting an action.

        Args:
            action_id: Action being exited
            output_type: Which output was taken
        """
        if self.current_path and self.current_path[-1].action_id == action_id:
            self.current_path[-1].exit_output = output_type

    def get_route_history(self) -> list[RouteRecord]:
        """
        Get complete routing history.

        Returns:
            List of all route records in order
        """
        return self.records.copy()

    def get_execution_path(self) -> list[str]:
        """
        Get execution path as list of action IDs.

        Returns:
            List of action IDs in execution order
        """
        return [segment.action_id for segment in self.current_path]

    def get_path_with_outputs(self) -> list[tuple[str, str, str | None]]:
        """
        Get execution path with output information.

        Returns:
            List of tuples (action_id, entry_output, exit_output)
        """
        return [(seg.action_id, seg.entry_output, seg.exit_output) for seg in self.current_path]

    def was_action_visited(self, action_id: str) -> bool:
        """
        Check if an action was visited during execution.

        Args:
            action_id: Action to check

        Returns:
            True if action was visited, False otherwise
        """
        return action_id in self.visited_actions

    def get_visit_count(self, action_id: str) -> int:
        """
        Get number of times an action was visited.

        Args:
            action_id: Action to check

        Returns:
            Visit count (0 if never visited)
        """
        return self.action_visit_count.get(action_id, 0)

    def get_output_usage(self, action_id: str) -> dict[str, int]:
        """
        Get usage count for each output type of an action.

        Args:
            action_id: Action to check

        Returns:
            Dictionary mapping output types to usage counts
        """
        return dict(self.output_usage.get(action_id, {}))

    def get_unvisited_actions(self, all_action_ids: list[str]) -> list[str]:
        """
        Get actions that were never visited.

        Args:
            all_action_ids: List of all action IDs in workflow

        Returns:
            List of unvisited action IDs
        """
        return [aid for aid in all_action_ids if aid not in self.visited_actions]

    def detect_loops(self) -> list[list[str]]:
        """
        Detect execution loops in the path.

        Returns:
            List of loops, where each loop is a list of action IDs
        """
        loops = []
        seen_positions: dict[str, list[int]] = defaultdict(list)

        # Track positions of each action in path
        for i, segment in enumerate(self.current_path):
            seen_positions[segment.action_id].append(i)

        # Find loops (action appears multiple times)
        for _action_id, positions in seen_positions.items():
            if len(positions) > 1:
                # Extract loop segments
                for i in range(len(positions) - 1):
                    start = positions[i]
                    end = positions[i + 1]
                    loop = [seg.action_id for seg in self.current_path[start : end + 1]]
                    loops.append(loop)

        return loops

    def get_branch_decisions(self) -> list[tuple[str, str]]:
        """
        Get all branching decisions (IF/SWITCH).

        Returns:
            List of tuples (action_id, branch_taken)
        """
        decisions = []

        for record in self.records:
            if record.output_type in ["true", "false"] or record.output_type.startswith("case_"):
                decisions.append((record.from_action, record.output_type))

        return decisions

    def get_error_routes(self) -> list[RouteRecord]:
        """
        Get all error routes taken.

        Returns:
            List of route records with 'error' output type
        """
        return [r for r in self.records if r.output_type == "error"]

    def finalize(self) -> None:
        """Mark execution as complete."""
        self.end_time = datetime.now()

    def get_execution_duration(self) -> float | None:
        """
        Get total execution duration in seconds.

        Returns:
            Duration in seconds, or None if not finalized
        """
        if self.end_time is None:
            return None

        return (self.end_time - self.start_time).total_seconds()

    def get_statistics(self) -> dict[str, any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with various statistics
        """
        stats = {
            "total_routes": len(self.records),
            "unique_actions_visited": len(self.visited_actions),
            "total_action_executions": sum(self.action_visit_count.values()),
            "path_length": len(self.current_path),
            "loops_detected": len(self.detect_loops()),
            "branch_decisions": len(self.get_branch_decisions()),
            "error_routes": len(self.get_error_routes()),
            "start_time": self.start_time.isoformat(),
        }

        if self.end_time:
            stats["end_time"] = self.end_time.isoformat()
            stats["duration_seconds"] = self.get_execution_duration()

        # Most visited action
        if self.action_visit_count:
            most_visited = max(self.action_visit_count.items(), key=lambda x: x[1])
            stats["most_visited_action"] = most_visited[0]
            stats["most_visited_count"] = most_visited[1]

        return stats

    def get_visual_path(self) -> str:
        """
        Get visual representation of execution path.

        Returns:
            String showing path with arrows and output types
        """
        if not self.current_path:
            return "(empty path)"

        parts = []
        for i, segment in enumerate(self.current_path):
            if i == 0:
                parts.append(f"[{segment.action_id}]")
            else:
                entry = segment.entry_output or "?"
                parts.append(f"--({entry})-> [{segment.action_id}]")

        return " ".join(parts)

    def get_route_graph(self) -> dict[str, list[tuple[str, str]]]:
        """
        Get routing graph structure.

        Returns:
            Dictionary mapping source action to list of (target, output_type) tuples
        """
        graph: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for record in self.records:
            graph[record.from_action].append((record.to_action, record.output_type))

        return dict(graph)

    def to_dict(self) -> dict:
        """
        Convert context to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "records": [
                {
                    "from": r.from_action,
                    "to": r.to_action,
                    "output": r.output_type,
                    "timestamp": r.timestamp.isoformat(),
                    "input_index": r.input_index,
                }
                for r in self.records
            ],
            "path": self.get_execution_path(),
            "statistics": self.get_statistics(),
            "visual_path": self.get_visual_path(),
        }

    def __str__(self) -> str:
        """String representation of routing context."""
        stats = self.get_statistics()
        return (
            f"RoutingContext("
            f"routes={stats['total_routes']}, "
            f"actions={stats['unique_actions_visited']}, "
            f"path_length={stats['path_length']}"
            f")"
        )
