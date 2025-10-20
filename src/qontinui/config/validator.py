"""
Validation logic for Qontinui action configurations.

This module provides utilities for validating action configurations,
checking action sequences, and ensuring data integrity.
"""

from typing import Any

from pydantic import ValidationError

from .schema import ACTION_CONFIG_MAP, Action, get_typed_config


class ActionValidationError(Exception):
    """Raised when action validation fails."""

    pass


class ActionValidator:
    """Validator for action configurations."""

    def __init__(self):
        """Initialize the validator."""
        self.known_action_types = set(ACTION_CONFIG_MAP.keys())

    def validate_action(self, action_data: dict[str, Any]) -> Action:
        """
        Validate a single action.

        Args:
            action_data: Raw action data dictionary

        Returns:
            Validated Action model

        Raises:
            ActionValidationError: If validation fails
        """
        try:
            action = Action.model_validate(action_data)
        except ValidationError as e:
            raise ActionValidationError(f"Invalid action structure: {e}") from e

        # Validate action type
        if action.type not in self.known_action_types:
            raise ActionValidationError(
                f"Unknown action type: {action.type}. "
                f"Known types: {sorted(self.known_action_types)}"
            )

        # Validate type-specific config
        try:
            get_typed_config(action)
        except ValidationError as e:
            raise ActionValidationError(
                f"Invalid config for action type '{action.type}': {e}"
            ) from e
        except ValueError as e:
            raise ActionValidationError(str(e)) from e

        return action

    def validate_actions(self, actions_data: list[dict[str, Any]]) -> list[Action]:
        """
        Validate a list of actions.

        Args:
            actions_data: List of raw action data dictionaries

        Returns:
            List of validated Action models

        Raises:
            ActionValidationError: If any validation fails
        """
        validated_actions = []
        errors = []

        for idx, action_data in enumerate(actions_data):
            try:
                action = self.validate_action(action_data)
                validated_actions.append(action)
            except ActionValidationError as e:
                errors.append(f"Action {idx} (id: {action_data.get('id', 'unknown')}): {e}")

        if errors:
            raise ActionValidationError(
                f"Validation failed for {len(errors)} action(s):\n" + "\n".join(errors)
            )

        return validated_actions

    def validate_action_sequence(
        self, actions: list[Action], check_references: bool = True
    ) -> list[str]:
        """
        Validate a sequence of actions for consistency.

        Args:
            actions: List of actions to validate
            check_references: Whether to check action ID references

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        action_ids = {action.id for action in actions}

        for action in actions:
            # Check for control flow action references
            if check_references:
                referenced_ids = self._get_referenced_action_ids(action)
                for ref_id in referenced_ids:
                    if ref_id not in action_ids:
                        warnings.append(
                            f"Action '{action.id}' references unknown action ID: '{ref_id}'"
                        )

            # Check for logical issues
            if action.type == "LOOP":
                config = get_typed_config(action)
                if config.loop_type == "FOR" and not config.iterations:
                    warnings.append(f"FOR loop action '{action.id}' has no iterations specified")
                elif config.loop_type == "WHILE" and not config.condition:
                    warnings.append(f"WHILE loop action '{action.id}' has no condition specified")
                elif config.loop_type == "FOREACH" and not config.collection:
                    warnings.append(
                        f"FOREACH loop action '{action.id}' has no collection specified"
                    )

            elif action.type == "TYPE":
                config = get_typed_config(action)
                if not config.text and not config.text_source:
                    warnings.append(
                        f"TYPE action '{action.id}' has no text or text_source specified"
                    )

            elif action.type == "WAIT":
                config = get_typed_config(action)
                if config.wait_for == "time" and not config.duration:
                    warnings.append(
                        f"WAIT action '{action.id}' wait_for='time' but no duration specified"
                    )
                elif config.wait_for == "target" and not config.target:
                    warnings.append(
                        f"WAIT action '{action.id}' wait_for='target' but no target specified"
                    )
                elif config.wait_for == "state" and not config.state_id:
                    warnings.append(
                        f"WAIT action '{action.id}' wait_for='state' but no state_id specified"
                    )
                elif config.wait_for == "condition" and not config.condition:
                    warnings.append(
                        f"WAIT action '{action.id}' wait_for='condition' but no condition specified"
                    )

        return warnings

    def _get_referenced_action_ids(self, action: Action) -> set[str]:
        """
        Extract action IDs referenced by control flow actions.

        Args:
            action: Action to extract references from

        Returns:
            Set of referenced action IDs
        """
        referenced_ids = set()

        try:
            config = get_typed_config(action)

            # Extract references based on action type
            if action.type == "IF":
                referenced_ids.update(config.then_actions)
                if config.else_actions:
                    referenced_ids.update(config.else_actions)

            elif action.type == "LOOP":
                referenced_ids.update(config.actions)

            elif action.type == "SWITCH":
                for case in config.cases:
                    referenced_ids.update(case.actions)
                if config.default_actions:
                    referenced_ids.update(config.default_actions)

            elif action.type == "TRY_CATCH":
                referenced_ids.update(config.try_actions)
                if config.catch_actions:
                    referenced_ids.update(config.catch_actions)
                if config.finally_actions:
                    referenced_ids.update(config.finally_actions)

        except Exception:
            # If config parsing fails, skip reference checking
            pass

        return referenced_ids

    def check_circular_references(
        self, actions: list[Action], start_action_id: str
    ) -> list[str] | None:
        """
        Check for circular references in action sequences.

        Args:
            actions: List of all actions
            start_action_id: ID of action to start checking from

        Returns:
            List representing the circular path if found, None otherwise
        """
        action_map = {action.id: action for action in actions}
        visited = set()
        path = []

        def dfs(action_id: str) -> list[str] | None:
            if action_id in visited:
                # Found a cycle
                cycle_start = path.index(action_id)
                return path[cycle_start:] + [action_id]

            if action_id not in action_map:
                return None

            visited.add(action_id)
            path.append(action_id)

            action = action_map[action_id]
            referenced_ids = self._get_referenced_action_ids(action)

            for ref_id in referenced_ids:
                cycle = dfs(ref_id)
                if cycle:
                    return cycle

            path.pop()
            visited.remove(action_id)
            return None

        return dfs(start_action_id)


def validate_action(action_data: dict[str, Any]) -> Action:
    """
    Convenience function to validate a single action.

    Args:
        action_data: Raw action data dictionary

    Returns:
        Validated Action model

    Raises:
        ActionValidationError: If validation fails
    """
    validator = ActionValidator()
    return validator.validate_action(action_data)


def validate_actions(actions_data: list[dict[str, Any]]) -> list[Action]:
    """
    Convenience function to validate a list of actions.

    Args:
        actions_data: List of raw action data dictionaries

    Returns:
        List of validated Action models

    Raises:
        ActionValidationError: If any validation fails
    """
    validator = ActionValidator()
    return validator.validate_actions(actions_data)


def validate_action_sequence(actions: list[Action], check_references: bool = True) -> list[str]:
    """
    Convenience function to validate an action sequence.

    Args:
        actions: List of actions to validate
        check_references: Whether to check action ID references

    Returns:
        List of warning messages (empty if no issues)
    """
    validator = ActionValidator()
    return validator.validate_action_sequence(actions, check_references)
