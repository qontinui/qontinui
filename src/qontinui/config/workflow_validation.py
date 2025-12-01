"""
Workflow validation - graph format only.

Clean, focused validation for graph-based workflows.
Errors only - no warnings. Make it pass or fail.
"""

from pydantic import BaseModel

from .schema import Workflow


class ValidationError(BaseModel):
    """Details about a validation error."""

    type: str
    message: str
    action_id: str | None = None
    details: dict | None = None


class ValidationResult(BaseModel):
    """Result of workflow validation."""

    valid: bool
    errors: list[ValidationError]

    def add_error(
        self,
        error_type: str,
        message: str,
        action_id: str | None = None,
        details: dict | None = None,
    ):
        """Add an error to the validation result."""
        self.errors.append(
            ValidationError(
                type=error_type, message=message, action_id=action_id, details=details
            )
        )
        self.valid = False


def validate_workflow(workflow: Workflow) -> ValidationResult:
    """
    Validate complete workflow structure.

    Performs all validation checks:
    - Connection validation
    - Position validation
    - Cycle detection
    - Orphan detection
    """
    result = ValidationResult(valid=True, errors=[])

    # Format must be 'graph'
    if workflow.format != "graph":
        result.add_error(
            "invalid_format",
            f"Workflow format must be 'graph', got '{workflow.format}'",
        )

    # Must have connections
    if workflow.connections is None:
        result.add_error(
            "missing_connections", "Graph format workflows must have connections field"
        )
        return result  # Can't continue without connections

    # Validate all graph-specific aspects
    validate_connections(workflow, result)
    validate_positions(workflow, result)

    # Detect cycles
    if detect_cycles(workflow):
        result.add_error("cycle_detected", "Workflow contains a cycle (infinite loop)")

    # Detect orphans (error in our clean design)
    orphans = detect_orphans(workflow)
    if orphans:
        result.add_error(
            "orphaned_actions",
            f"Found {len(orphans)} orphaned action(s) not connected to workflow",
            details={"orphaned_actions": orphans},
        )

    return result


def validate_connections(
    workflow: Workflow, result: ValidationResult | None = None
) -> ValidationResult:
    """
    Validate all connections reference valid actions.

    Checks:
    - All target action IDs exist in workflow
    - Connection types are valid
    - Indices are reasonable
    """
    if result is None:
        result = ValidationResult(valid=True, errors=[])

    if workflow.connections is None:
        return result

    # Build set of valid action IDs
    action_ids = {action.id for action in workflow.actions}

    # Validate each connection
    for source_id, connection_types in workflow.connections.root.items():
        # Check source action exists
        if source_id not in action_ids:
            result.add_error(
                "invalid_source_action",
                f"Connection source action '{source_id}' does not exist in workflow",
                action_id=source_id,
            )
            continue

        for conn_type, output_list in connection_types.items():
            for output_idx, connections in enumerate(output_list):
                for conn in connections:
                    # Check target action exists
                    if conn.action not in action_ids:
                        result.add_error(
                            "invalid_target_action",
                            f"Connection target action '{conn.action}' does not exist in workflow",
                            action_id=source_id,
                            details={
                                "target_action": conn.action,
                                "connection_type": conn_type,
                                "output_index": output_idx,
                            },
                        )

                    # Validate connection type
                    valid_types = ["main", "error", "success", "true", "false"]
                    if not (conn_type in valid_types or conn_type.startswith("case_")):
                        result.add_error(
                            "unusual_connection_type",
                            f"Unusual connection type '{conn_type}'",
                            action_id=source_id,
                        )

    return result


def validate_positions(
    workflow: Workflow, result: ValidationResult | None = None
) -> ValidationResult:
    """
    Validate positions are present and reasonable.

    Checks:
    - All actions have positions (for graph format)
    - Positions are valid tuples
    - Positions are finite numbers
    """
    if result is None:
        result = ValidationResult(valid=True, errors=[])

    for action in workflow.actions:
        if action.position is None:
            result.add_error(
                "missing_position",
                f"Action '{action.id}' is missing position (required for graph format)",
                action_id=action.id,
            )
            continue

        x, y = action.position

        # Check non-negative
        if x < 0 or y < 0:
            result.add_error(
                "negative_position",
                f"Action '{action.id}' has negative position: ({x}, {y})",
                action_id=action.id,
                details={"position": [x, y]},
            )

        # Check finite values
        if not (isinstance(x, int | float) and isinstance(y, int | float)):
            result.add_error(
                "invalid_position_type",
                f"Action '{action.id}' has non-numeric position values",
                action_id=action.id,
                details={"position": [x, y]},
            )

    return result


def detect_cycles(workflow: Workflow) -> bool:
    """
    Detect cycles in workflow graph using DFS.

    A cycle means an action can reach itself through connections,
    which could cause infinite loops during execution.
    """
    if workflow.connections is None:
        return False

    # Build adjacency list
    graph: dict[str, list[str]] = {action.id: [] for action in workflow.actions}

    for source_id, connection_types in workflow.connections.root.items():
        for connections_list in connection_types.values():
            for connections in connections_list:
                for conn in connections:
                    if conn.action in graph:
                        graph[source_id].append(conn.action)

    # DFS-based cycle detection
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def has_cycle_util(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if has_cycle_util(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for action_id in graph:
        if action_id not in visited:
            if has_cycle_util(action_id):
                return True

    return False


def detect_orphans(workflow: Workflow) -> list[str]:
    """
    Find actions not connected to anything in a graph workflow.

    An orphaned action has no incoming AND no outgoing connections.
    """
    if workflow.connections is None:
        return []

    # Track actions with incoming/outgoing connections
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()

    for source_id, connection_types in workflow.connections.root.items():
        has_outgoing.add(source_id)
        for connections_list in connection_types.values():
            for connections in connections_list:
                for conn in connections:
                    has_incoming.add(conn.action)

    # Find truly orphaned actions (no incoming AND no outgoing)
    orphans = []
    for action in workflow.actions:
        if action.id not in has_incoming and action.id not in has_outgoing:
            orphans.append(action.id)

    return orphans
