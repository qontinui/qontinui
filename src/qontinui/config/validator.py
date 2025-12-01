"""
Validation logic for Qontinui action configurations.

This module provides utilities for validating action configurations,
checking action sequences, and ensuring data integrity.
"""

from typing import Any, cast

from pydantic import ValidationError

from .models.control_flow import (
    IfActionConfig,
    LoopActionConfig,
    SwitchActionConfig,
    TryCatchActionConfig,
)
from .models.find_actions import WaitActionConfig
from .models.keyboard_actions import TypeActionConfig
from .schema import ACTION_CONFIG_MAP, Action, get_typed_config


class ActionValidationError(Exception):
    """Raised when action validation fails."""

    pass


class ActionValidator:
    """Validator for action configurations."""

    def __init__(self) -> None:
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
                errors.append(
                    f"Action {idx} (id: {action_data.get('id', 'unknown')}): {e}"
                )

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
                loop_config = cast(LoopActionConfig, get_typed_config(action))
                if loop_config.loop_type == "FOR" and not loop_config.iterations:
                    warnings.append(
                        f"FOR loop action '{action.id}' has no iterations specified"
                    )
                elif loop_config.loop_type == "WHILE" and not loop_config.condition:
                    warnings.append(
                        f"WHILE loop action '{action.id}' has no condition specified"
                    )
                elif loop_config.loop_type == "FOREACH" and not loop_config.collection:
                    warnings.append(
                        f"FOREACH loop action '{action.id}' has no collection specified"
                    )

            elif action.type == "TYPE":
                type_config = cast(TypeActionConfig, get_typed_config(action))
                if not type_config.text and not type_config.text_source:
                    warnings.append(
                        f"TYPE action '{action.id}' has no text or text_source specified"
                    )

            elif action.type == "WAIT":
                wait_config = cast(WaitActionConfig, get_typed_config(action))
                if wait_config.wait_for == "time" and not wait_config.duration:
                    warnings.append(
                        f"WAIT action '{action.id}' wait_for='time' but no duration specified"
                    )
                elif wait_config.wait_for == "target" and not wait_config.target:
                    warnings.append(
                        f"WAIT action '{action.id}' wait_for='target' but no target specified"
                    )
                elif wait_config.wait_for == "state" and not wait_config.state_id:
                    warnings.append(
                        f"WAIT action '{action.id}' wait_for='state' but no state_id specified"
                    )
                elif wait_config.wait_for == "condition" and not wait_config.condition:
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
            # Extract references based on action type
            if action.type == "IF":
                if_config = cast(IfActionConfig, get_typed_config(action))
                referenced_ids.update(if_config.then_actions)
                if if_config.else_actions:
                    referenced_ids.update(if_config.else_actions)

            elif action.type == "LOOP":
                loop_config = cast(LoopActionConfig, get_typed_config(action))
                referenced_ids.update(loop_config.actions)

            elif action.type == "SWITCH":
                switch_config = cast(SwitchActionConfig, get_typed_config(action))
                for case in switch_config.cases:
                    referenced_ids.update(case.actions)
                if switch_config.default_actions:
                    referenced_ids.update(switch_config.default_actions)

            elif action.type == "TRY_CATCH":
                try_catch_config = cast(TryCatchActionConfig, get_typed_config(action))
                referenced_ids.update(try_catch_config.try_actions)
                if try_catch_config.catch_actions:
                    referenced_ids.update(try_catch_config.catch_actions)
                if try_catch_config.finally_actions:
                    referenced_ids.update(try_catch_config.finally_actions)

        except (ValidationError, AttributeError, KeyError):
            # If config parsing fails, skip reference checking for this action
            # This can happen with malformed or incomplete action configs
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
        path: list[str] = []

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


def validate_action_sequence(
    actions: list[Action], check_references: bool = True
) -> list[str]:
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
