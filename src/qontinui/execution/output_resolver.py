"""
Output resolution for action execution results.

Maps action execution results to output types (connection types) that determine
the next action(s) to execute in the workflow graph.

Clean, focused design for graph-based workflow execution.
"""

from typing import Any

from ..config.schema import Action, get_typed_config


class OutputResolver:
    """
    Resolves action execution results to output types for routing.

    Different action types have different output patterns:
    - IF: Returns 'true' or 'false' based on condition
    - LOOP: Returns 'loop' to continue or 'main' to exit
    - SWITCH: Returns 'case_N' based on switch value or 'default'
    - TRY_CATCH: Returns 'main' on success or 'error' on exception
    - Standard actions: Return 'main' for normal flow, 'error' on failure
    """

    def resolve(self, action: Action, result: dict[str, Any]) -> str:
        """
        Resolve execution result to output type.

        Args:
            action: The executed action
            result: Execution result dictionary

        Returns:
            Output type string (e.g., 'main', 'true', 'false', 'error', 'case_0')

        Raises:
            ValueError: If action type is unknown or result is invalid
        """
        # Check for error first (applies to all actions)
        if result.get("error") or result.get("success") is False:
            return "error"

        # Route based on action type
        if action.type == "IF":
            return self.resolve_if_output(result)

        elif action.type == "LOOP":
            return self.resolve_loop_output(result)

        elif action.type == "SWITCH":
            return self.resolve_switch_output(action, result)

        elif action.type == "TRY_CATCH":
            return self.resolve_try_catch_output(result)

        else:
            # Standard actions use 'main' output
            return "main"

    def resolve_if_output(self, result: dict[str, Any]) -> str:
        """
        Resolve IF action output.

        Args:
            result: Execution result with 'condition_result' field

        Returns:
            'true' if condition is true, 'false' otherwise

        Raises:
            ValueError: If result doesn't contain condition_result
        """
        if "condition_result" not in result:
            raise ValueError("IF action result must contain 'condition_result' field")

        condition = result["condition_result"]
        return "true" if condition else "false"

    def resolve_loop_output(self, result: dict[str, Any]) -> str:
        """
        Resolve LOOP action output.

        Args:
            result: Execution result with 'continue_loop' field

        Returns:
            'loop' to continue looping, 'main' to exit loop

        Raises:
            ValueError: If result doesn't contain continue_loop
        """
        if "continue_loop" not in result:
            raise ValueError("LOOP action result must contain 'continue_loop' field")

        continue_loop = result["continue_loop"]
        return "loop" if continue_loop else "main"

    def resolve_switch_output(self, action: Action, result: dict[str, Any]) -> str:
        """
        Resolve SWITCH action output.

        Args:
            action: The SWITCH action
            result: Execution result with 'case_index' field

        Returns:
            'case_N' where N is the matched case index, or 'default'

        Raises:
            ValueError: If result doesn't contain case_index
        """
        if "case_index" not in result:
            raise ValueError("SWITCH action result must contain 'case_index' field")

        case_index = result["case_index"]

        if case_index is None:
            return "default"

        return f"case_{case_index}"

    def resolve_try_catch_output(self, result: dict[str, Any]) -> str:
        """
        Resolve TRY_CATCH action output.

        Args:
            result: Execution result with optional 'error' field

        Returns:
            'error' if exception occurred, 'main' on success
        """
        # Error field indicates exception was caught
        if result.get("error") or result.get("exception"):
            return "error"

        return "main"

    def get_valid_outputs(self, action: Action) -> list[str]:
        """
        Get list of valid output types for an action.

        Args:
            action: The action to check

        Returns:
            List of valid output type strings
        """
        # All actions can have 'error' output
        base_outputs = ["error"]

        if action.type == "IF":
            return base_outputs + ["true", "false"]

        elif action.type == "LOOP":
            return base_outputs + ["loop", "main"]

        elif action.type == "SWITCH":
            config = get_typed_config(action)
            cases = config.cases if hasattr(config, "cases") else []
            outputs = base_outputs + [f"case_{i}" for i in range(len(cases))]
            outputs.append("default")
            return outputs

        elif action.type == "TRY_CATCH":
            return ["main", "error"]  # No general error for try_catch, it handles it

        else:
            # Standard actions have main and error
            return base_outputs + ["main"]

    def validate_output_exists(self, action: Action, output_type: str) -> bool:
        """
        Check if an output type is valid for an action.

        Args:
            action: The action to validate
            output_type: The output type to check

        Returns:
            True if output type is valid, False otherwise
        """
        valid_outputs = self.get_valid_outputs(action)
        return output_type in valid_outputs

    def get_output_description(self, action: Action, output_type: str) -> str:
        """
        Get human-readable description of an output type.

        Args:
            action: The action
            output_type: The output type

        Returns:
            Description string
        """
        descriptions = {
            "main": "Normal execution flow",
            "error": "Error or failure occurred",
            "true": "Condition evaluated to true",
            "false": "Condition evaluated to false",
            "loop": "Continue looping",
            "default": "No case matched (default)",
        }

        # Check for case_N pattern
        if output_type.startswith("case_"):
            try:
                case_num = int(output_type.split("_")[1])
                return f"Switch case {case_num} matched"
            except (IndexError, ValueError):
                pass

        return descriptions.get(output_type, f"Output type: {output_type}")

    def get_expected_result_fields(self, action: Action) -> list[str]:
        """
        Get expected fields in execution result for an action type.

        Args:
            action: The action

        Returns:
            List of expected field names
        """
        base_fields = ["success", "error"]

        if action.type == "IF":
            return base_fields + ["condition_result"]

        elif action.type == "LOOP":
            return base_fields + ["continue_loop", "iteration", "max_iterations"]

        elif action.type == "SWITCH":
            return base_fields + ["case_index", "matched_value"]

        elif action.type == "TRY_CATCH":
            return base_fields + ["exception", "error_message"]

        else:
            return base_fields

    def validate_result_structure(
        self, action: Action, result: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """
        Validate that execution result has expected structure.

        Args:
            action: The executed action
            result: Execution result dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(result, dict):
            return False, f"Result must be a dictionary, got {type(result).__name__}"

        # For control flow actions, check required fields
        if action.type == "IF" and "condition_result" not in result:
            return False, "IF action result missing 'condition_result' field"

        if action.type == "LOOP" and "continue_loop" not in result:
            return False, "LOOP action result missing 'continue_loop' field"

        if action.type == "SWITCH" and "case_index" not in result:
            return False, "SWITCH action result missing 'case_index' field"

        return True, None


class OutputTypeValidator:
    """
    Validates output types against action configurations.

    Ensures connections in workflow use valid output types for their source actions.
    """

    def __init__(self, resolver: OutputResolver | None = None) -> None:
        """
        Initialize validator.

        Args:
            resolver: OutputResolver instance (creates new if not provided)
        """
        self.resolver = resolver or OutputResolver()

    def validate_connection_type(
        self, action: Action, connection_type: str
    ) -> tuple[bool, str | None]:
        """
        Validate that a connection type is valid for an action.

        Args:
            action: Source action
            connection_type: Type of connection (e.g., 'main', 'true', 'error')

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.resolver.validate_output_exists(action, connection_type):
            valid_types = self.resolver.get_valid_outputs(action)
            return False, (
                f"Invalid connection type '{connection_type}' for {action.type} action. "
                f"Valid types: {', '.join(valid_types)}"
            )

        return True, None

    def get_missing_outputs(self, action: Action, used_types: list[str]) -> list[str]:
        """
        Get output types that exist but aren't used in connections.

        Args:
            action: The action
            used_types: List of connection types used

        Returns:
            List of unused output types
        """
        valid_outputs = self.resolver.get_valid_outputs(action)
        return [out for out in valid_outputs if out not in used_types]

    def get_invalid_outputs(self, action: Action, used_types: list[str]) -> list[str]:
        """
        Get connection types that aren't valid for the action.

        Args:
            action: The action
            used_types: List of connection types used

        Returns:
            List of invalid connection types
        """
        valid_outputs = self.resolver.get_valid_outputs(action)
        return [out for out in used_types if out not in valid_outputs]
