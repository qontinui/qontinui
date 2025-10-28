"""
Connection resolver for graph-based workflow execution.

This module provides functionality for resolving connections between actions
in graph format workflows. It handles different output types (main, true/false,
loop, error, case_N) and validates connection existence.
"""

from ..config.schema import Action, Connection, Workflow


class ConnectionResolver:
    """
    Resolves connections between actions in graph workflows.

    This class handles the complexity of multi-output actions (IF, SWITCH, TRY_CATCH, LOOP)
    and provides validation for connection existence and integrity.
    """

    # Valid output types for different action types
    VALID_OUTPUT_TYPES = {
        "IF": {"true", "false", "error"},
        "SWITCH": {
            "case_0",
            "case_1",
            "case_2",
            "case_3",
            "case_4",
            "case_5",
            "case_6",
            "case_7",
            "case_8",
            "case_9",
            "error",
        },
        "LOOP": {"main", "loop", "error"},
        "TRY_CATCH": {"main", "error", "finally"},
        "default": {"main", "error"},
    }

    def __init__(self, workflow: Workflow) -> None:
        """
        Initialize the connection resolver.

        Args:
            workflow: The workflow containing actions and connections
        """
        self.workflow = workflow
        self.connections = workflow.connections
        self._action_map = {action.id: action for action in workflow.actions}
        self._validate_connections()

    def _validate_connections(self) -> None:
        """
        Validate that all connections reference existing actions.

        Raises:
            ValueError: If any connection references a non-existent action
        """
        for source_id, output_types in self.connections.root.items():
            if source_id not in self._action_map:
                raise ValueError(
                    f"Connection source action '{source_id}' does not exist in workflow"
                )

            for _output_type, connection_groups in output_types.items():
                for group in connection_groups:
                    for conn in group:
                        if conn.action not in self._action_map:
                            raise ValueError(
                                f"Connection target action '{conn.action}' does not exist in workflow"
                            )

    def resolve_output_connection(
        self, action_id: str, output_type: str, index: int = 0
    ) -> list[Connection]:
        """
        Resolve connections for a specific output of an action.

        Args:
            action_id: The source action ID
            output_type: The output type (main, true, false, loop, error, case_N)
            index: The output index (for actions with multiple outputs of same type)

        Returns:
            List of connections for this output

        Raises:
            ValueError: If action doesn't exist or output type is invalid
        """
        # Validate action exists
        if action_id not in self._action_map:
            raise ValueError(f"Action '{action_id}' does not exist in workflow")

        action = self._action_map[action_id]

        # Validate output type for this action
        valid_types = self._get_valid_output_types(action)
        if output_type not in valid_types:
            raise ValueError(
                f"Invalid output type '{output_type}' for action '{action_id}' "
                f"of type '{action.type}'. Valid types: {valid_types}"
            )

        # Get connections for this action and output type
        all_connections = self.connections.get_connections(action_id, output_type)

        # Return connections at the specified index
        if index < len(all_connections):
            return all_connections[index]
        return []

    def get_all_outputs(self, action_id: str) -> list[tuple[str, int]]:
        """
        Get all outputs for an action.

        Args:
            action_id: The action ID

        Returns:
            List of (output_type, index) tuples for all outputs
        """
        if action_id not in self._action_map:
            return []

        outputs = []
        action_connections = self.connections.get_all_connections(action_id)

        for output_type, connection_groups in action_connections.items():
            for index in range(len(connection_groups)):
                outputs.append((output_type, index))

        return outputs

    def _get_valid_output_types(self, action: Action) -> set[str]:
        """
        Get valid output types for an action based on its type.

        Args:
            action: The action to get valid output types for

        Returns:
            Set of valid output types
        """
        action_type = action.type

        if action_type in self.VALID_OUTPUT_TYPES:
            return self.VALID_OUTPUT_TYPES[action_type]

        # For SWITCH actions, we need to check the number of cases
        if action_type == "SWITCH":
            # Get case count from config if available
            case_count = len(action.config.get("cases", []))
            valid_types = {f"case_{i}" for i in range(case_count)}
            valid_types.add("error")
            return valid_types

        return self.VALID_OUTPUT_TYPES["default"]

    def has_output(self, action_id: str, output_type: str) -> bool:
        """
        Check if an action has a specific output type.

        Args:
            action_id: The action ID
            output_type: The output type to check

        Returns:
            True if the action has this output type
        """
        if action_id not in self._action_map:
            return False

        action = self._action_map[action_id]
        valid_types = self._get_valid_output_types(action)
        return output_type in valid_types

    def get_connected_actions(
        self, action_id: str, output_type: str = "main", index: int = 0
    ) -> list[Action]:
        """
        Get all actions connected to a specific output.

        Args:
            action_id: The source action ID
            output_type: The output type
            index: The output index

        Returns:
            List of connected actions
        """
        connections = self.resolve_output_connection(action_id, output_type, index)
        return [self._action_map[conn.action] for conn in connections]

    def get_action_by_id(self, action_id: str) -> Action | None:
        """
        Get an action by its ID.

        Args:
            action_id: The action ID

        Returns:
            The action, or None if not found
        """
        return self._action_map.get(action_id)

    def get_incoming_connections(self, action_id: str) -> list[tuple[str, str, int]]:
        """
        Get all incoming connections to an action.

        Args:
            action_id: The target action ID

        Returns:
            List of (source_action_id, output_type, output_index) tuples
        """
        incoming = []

        for source_id, output_types in self.connections.root.items():
            for output_type, connection_groups in output_types.items():
                for index, group in enumerate(connection_groups):
                    for conn in group:
                        if conn.action == action_id:
                            incoming.append((source_id, output_type, index))

        return incoming

    def has_incoming_connections(self, action_id: str) -> bool:
        """
        Check if an action has any incoming connections.

        Args:
            action_id: The action ID

        Returns:
            True if the action has incoming connections
        """
        return len(self.get_incoming_connections(action_id)) > 0

    def validate_output_exists(self, action_id: str, output_type: str) -> bool:
        """
        Validate that an action has a specific output with connections.

        Args:
            action_id: The action ID
            output_type: The output type

        Returns:
            True if the output exists and has connections
        """
        if not self.has_output(action_id, output_type):
            return False

        connections = self.connections.get_connections(action_id, output_type)
        return len(connections) > 0

    def get_output_count(self, action_id: str) -> int:
        """
        Get the total number of outputs for an action.

        Args:
            action_id: The action ID

        Returns:
            Total number of outputs
        """
        if action_id not in self._action_map:
            return 0

        return len(self.get_all_outputs(action_id))

    def is_multi_output_action(self, action_id: str) -> bool:
        """
        Check if an action has multiple outputs.

        Args:
            action_id: The action ID

        Returns:
            True if the action has more than one output type
        """
        if action_id not in self._action_map:
            return False

        action_connections = self.connections.get_all_connections(action_id)
        return len(action_connections) > 1

    def get_branch_actions(self, action_id: str) -> dict[str, list[Action]]:
        """
        Get all branching actions for multi-output actions.

        Useful for IF, SWITCH, TRY_CATCH actions.

        Args:
            action_id: The action ID

        Returns:
            Dictionary mapping output type to list of connected actions
        """
        if action_id not in self._action_map:
            return {}

        branches = {}
        action_connections = self.connections.get_all_connections(action_id)

        for output_type, connection_groups in action_connections.items():
            branch_actions = []
            for group in connection_groups:
                for conn in group:
                    branch_actions.append(self._action_map[conn.action])
            branches[output_type] = branch_actions

        return branches
