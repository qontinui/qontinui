"""Keyboard action executor for the command pattern refactoring.

This module handles all keyboard-related actions: KEY_DOWN, KEY_UP, KEY_PRESS, and TYPE.
"""

import logging
from typing import Any

from ..config.schema import (
    Action,
    KeyDownActionConfig,
    KeyPressActionConfig,
    KeyUpActionConfig,
    TypeActionConfig,
)
from ..exceptions import ActionExecutionError
from .base import ActionExecutorBase
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class KeyboardActionExecutor(ActionExecutorBase):
    """Executor for keyboard actions.

    Handles:
        - KEY_DOWN: Press and hold a key
        - KEY_UP: Release a key
        - KEY_PRESS: Press and release key(s)
        - TYPE: Type text string with optional variable resolution

    Example:
        context = ExecutionContext(...)
        executor = KeyboardActionExecutor(context)

        # Press and hold Ctrl
        action = Action(type="KEY_DOWN", config={"keys": ["ctrl"]})
        executor.execute(action, KeyDownActionConfig(keys=["ctrl"]))

        # Type text
        action = Action(type="TYPE", config={"text": "Hello World"})
        executor.execute(action, TypeActionConfig(text="Hello World"))
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of keyboard action types this executor handles.

        Returns:
            List containing: KEY_DOWN, KEY_UP, KEY_PRESS, TYPE
        """
        return ["KEY_DOWN", "KEY_UP", "KEY_PRESS", "TYPE"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute a keyboard action with validated configuration.

        Args:
            action: Pydantic Action model with type, config, etc.
            typed_config: Type-specific validated configuration object

        Returns:
            True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        action_type = action.type

        try:
            if action_type == "KEY_DOWN":
                return self._execute_key_down(action, typed_config)
            elif action_type == "KEY_UP":
                return self._execute_key_up(action, typed_config)
            elif action_type == "KEY_PRESS":
                return self._execute_key_press(action, typed_config)
            elif action_type == "TYPE":
                return self._execute_type(action, typed_config)
            else:
                raise ActionExecutionError(
                    action_type=action_type,
                    reason=f"Unsupported action type: {action_type}",
                )

        except ActionExecutionError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing {action_type}: {e}", exc_info=True)
            raise ActionExecutionError(
                action_type=action_type,
                reason=f"Unexpected error: {e}",
            ) from e

    def _execute_key_down(self, action: Action, typed_config: KeyDownActionConfig | None) -> bool:
        """Execute KEY_DOWN action - press and hold key.

        This is a pure action that only presses the key down.
        The key remains pressed until KEY_UP is called.

        Args:
            action: Action model
            typed_config: Validated KeyDownActionConfig

        Returns:
            True if successful

        Raises:
            ActionExecutionError: If required parameters missing
        """
        # Get key from typed config or fall back to action.config
        key = None
        if typed_config and typed_config.keys:
            key = typed_config.keys[0]  # Take first key
        elif "key" in action.config:
            key = action.config["key"]

        if not key:
            error_msg = "KEY_DOWN requires 'key' parameter"
            logger.error(error_msg)
            raise ActionExecutionError(
                action_type="KEY_DOWN",
                reason="Missing required parameter 'key'",
            )

        try:
            self.context.keyboard.down(key)
            logger.info(f"Key '{key}' pressed down")
            self._emit_action_success(action, {"key": key})
            return True

        except Exception as e:
            logger.error(f"Failed to press key '{key}': {e}")
            self._emit_action_failure(action, str(e), {"key": key})
            return False

    def _execute_key_up(self, action: Action, typed_config: KeyUpActionConfig | None) -> bool:
        """Execute KEY_UP action - release key.

        This is a pure action that only releases the key.

        Args:
            action: Action model
            typed_config: Validated KeyUpActionConfig

        Returns:
            True if successful

        Raises:
            ActionExecutionError: If required parameters missing
        """
        # Get key from typed config or fall back to action.config
        key = None
        if typed_config and typed_config.keys:
            key = typed_config.keys[0]  # Take first key
        elif "key" in action.config:
            key = action.config["key"]

        if not key:
            error_msg = "KEY_UP requires 'key' parameter"
            logger.error(error_msg)
            raise ActionExecutionError(
                action_type="KEY_UP",
                reason="Missing required parameter 'key'",
            )

        try:
            self.context.keyboard.up(key)
            logger.info(f"Key '{key}' released")
            self._emit_action_success(action, {"key": key})
            return True

        except Exception as e:
            logger.error(f"Failed to release key '{key}': {e}")
            self._emit_action_failure(action, str(e), {"key": key})
            return False

    def _execute_key_press(self, action: Action, typed_config: KeyPressActionConfig | None) -> bool:
        """Execute KEY_PRESS action - press and release key(s).

        This is a pure action that presses and immediately releases key(s).
        Equivalent to KEY_DOWN + KEY_UP for each key.

        Args:
            action: Action model
            typed_config: Validated KeyPressActionConfig

        Returns:
            True if successful
        """
        # Get keys from typed config or fall back to action.config
        keys = []
        if typed_config and typed_config.keys:
            keys = typed_config.keys
        elif "keys" in action.config:
            keys = action.config["keys"]
        elif "key" in action.config:
            keys = [action.config["key"]]

        if not keys:
            logger.warning("KEY_PRESS action has no keys specified, skipping")
            return True

        try:
            for key in keys:
                self.context.keyboard.press(key)
                logger.info(f"Key '{key}' pressed and released")

            self._emit_action_success(action, {"keys": keys})
            return True

        except Exception as e:
            logger.error(f"Failed to press keys {keys}: {e}")
            self._emit_action_failure(action, str(e), {"keys": keys})
            return False

    def _execute_type(self, action: Action, typed_config: TypeActionConfig | None) -> bool:
        """Execute TYPE action with variable resolution and state string lookup.

        Supports:
        - Direct text: {"text": "Hello World"}
        - Variable resolution: Variables from context are automatically resolved
        - State strings: {"textSource": {"stateId": "...", "stringIds": [...]}}

        Args:
            action: Action model
            typed_config: Validated TypeActionConfig

        Returns:
            True if text was typed successfully

        Raises:
            ActionExecutionError: If no text source provided
        """
        logger.debug("Executing TYPE action")

        if not typed_config:
            error_msg = "TYPE action requires valid TypeActionConfig"
            logger.error(error_msg)
            raise ActionExecutionError(
                action_type="TYPE",
                reason="Missing or invalid configuration",
            )

        text = ""

        # Get text directly or from text_source
        if typed_config.text:
            text = typed_config.text
            logger.debug(f"Using direct text: '{text}'")

        elif typed_config.text_source:
            # Get text from state string
            state_id = typed_config.text_source.state_id
            string_ids = typed_config.text_source.string_ids

            logger.debug(f"Looking for state string: state_id={state_id}, string_ids={string_ids}")

            if state_id and string_ids:
                # Access state_map through context.config
                state_map = getattr(self.context.config, "state_map", {})

                if state_id in state_map:
                    state = state_map[state_id]
                    state_strings = getattr(state, "state_strings", [])
                    logger.debug(
                        f"State strings in '{state_id}': {[(s.id, s.value) for s in state_strings]}"
                    )

                    # Find the string in the state
                    for state_string in state_strings:
                        if state_string.id in string_ids:
                            text = state_string.value
                            logger.debug(f"Found matching string: '{text}'")
                            break

                    if not text:
                        error_msg = f"No matching string found for IDs: {string_ids}"
                        logger.error(error_msg)
                        self._emit_action_failure(action, error_msg)
                        return False
                else:
                    error_msg = f"State '{state_id}' not found"
                    logger.error(error_msg)
                    self._emit_action_failure(action, error_msg)
                    return False
            else:
                error_msg = "text_source requires both state_id and string_ids"
                logger.error(error_msg)
                self._emit_action_failure(action, error_msg)
                return False

        # Resolve variables in text if variable context is available
        if text and self.context.variable_context:
            try:
                # Variables in the format ${varName} are automatically resolved
                # by the variable context if the text contains them
                import re

                var_pattern = r"\$\{([^}]+)\}"
                matches = re.findall(var_pattern, text)

                for var_name in matches:
                    var_value = self.context.variable_context.get(var_name)
                    if var_value is not None:
                        text = text.replace(f"${{{var_name}}}", str(var_value))
                        logger.debug(f"Resolved variable '{var_name}' to '{var_value}'")
                    else:
                        logger.warning(f"Variable '{var_name}' not found in context")

            except Exception as e:
                logger.warning(f"Error resolving variables: {e}")
                # Continue with unresolved text

        # Type the text if we have it
        if text:
            try:
                self.context.keyboard.type(text)
                logger.info(f"Successfully typed: '{text}'")

                # Emit TEXT_TYPED event for runner/frontend
                import sys

                print(
                    f"[KEYBOARD_EXECUTOR] About to emit TEXT_TYPED event for text: '{text}'",
                    file=sys.stderr,
                    flush=True,
                )
                from ..reporting.events import EventType, emit_event

                emit_event(EventType.TEXT_TYPED, {"text": text, "character_count": len(text)})
                print(
                    "[KEYBOARD_EXECUTOR] TEXT_TYPED event emitted successfully",
                    file=sys.stderr,
                    flush=True,
                )

                self._emit_action_success(action, {"text": text, "length": len(text)})
                return True

            except Exception as e:
                logger.error(f"Failed to type text '{text}': {e}")
                self._emit_action_failure(action, str(e), {"text": text})
                return False

        error_msg = "TYPE action failed - no text to type"
        logger.error(error_msg)
        self._emit_action_failure(action, error_msg)
        return False
