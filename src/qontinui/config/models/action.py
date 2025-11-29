"""
Core action model and configuration mapping.

This module provides the main Action model that wraps all action types,
along with utilities for type-safe configuration access.
"""

from typing import Any

from pydantic import BaseModel

from .code_actions import (
    CodeBlockActionConfig,
    CustomFunctionActionConfig,
)
from .control_flow import (
    BreakActionConfig,
    ContinueActionConfig,
    IfActionConfig,
    LoopActionConfig,
    SwitchActionConfig,
    TryCatchActionConfig,
)
from .data_operations import (
    FilterActionConfig,
    GetVariableActionConfig,
    MapActionConfig,
    MathOperationActionConfig,
    ReduceActionConfig,
    SetVariableActionConfig,
    SortActionConfig,
    StringOperationActionConfig,
)
from .execution import BaseActionSettings, ExecutionSettings
from .find_actions import (
    ExistsActionConfig,
    FindActionConfig,
    FindStateImageActionConfig,
    VanishActionConfig,
    WaitActionConfig,
)
from .keyboard_actions import (
    HotkeyActionConfig,
    KeyDownActionConfig,
    KeyPressActionConfig,
    KeyUpActionConfig,
    TypeActionConfig,
)
from .mouse_actions import (
    ClickActionConfig,
    DragActionConfig,
    HighlightActionConfig,
    MouseDownActionConfig,
    MouseMoveActionConfig,
    MouseUpActionConfig,
    ScrollActionConfig,
)
from .state_actions import (
    GoToStateActionConfig,
    RunWorkflowActionConfig,
    ScreenshotActionConfig,
)


class Action(BaseModel):
    """
    Main action model - supports both graph and sequential formats.

    Position is optional - required for graph format, not needed for sequential.
    Config can be parsed into type-specific models using get_typed_config().
    """

    id: str
    type: str
    name: str | None = None
    config: dict[str, Any]
    base: BaseActionSettings | None = None
    execution: ExecutionSettings | None = None
    position: tuple[int, int] | None = None  # Optional: only for graph format


# Mapping of action types to their config models
ACTION_CONFIG_MAP = {
    # Find actions
    "FIND": FindActionConfig,
    "FIND_STATE_IMAGE": FindStateImageActionConfig,
    "EXISTS": ExistsActionConfig,
    "VANISH": VanishActionConfig,
    "WAIT": WaitActionConfig,
    # Mouse actions
    "CLICK": ClickActionConfig,
    "MOUSE_MOVE": MouseMoveActionConfig,
    "MOUSE_DOWN": MouseDownActionConfig,
    "MOUSE_UP": MouseUpActionConfig,
    "DRAG": DragActionConfig,
    "SCROLL": ScrollActionConfig,
    "HIGHLIGHT": HighlightActionConfig,
    # Keyboard actions
    "TYPE": TypeActionConfig,
    "KEY_PRESS": KeyPressActionConfig,
    "KEY_DOWN": KeyDownActionConfig,
    "KEY_UP": KeyUpActionConfig,
    "HOTKEY": HotkeyActionConfig,
    # Control flow actions
    "IF": IfActionConfig,
    "LOOP": LoopActionConfig,
    "BREAK": BreakActionConfig,
    "CONTINUE": ContinueActionConfig,
    "SWITCH": SwitchActionConfig,
    "TRY_CATCH": TryCatchActionConfig,
    # Data actions
    "SET_VARIABLE": SetVariableActionConfig,
    "GET_VARIABLE": GetVariableActionConfig,
    "SORT": SortActionConfig,
    "FILTER": FilterActionConfig,
    "MAP": MapActionConfig,
    "REDUCE": ReduceActionConfig,
    "STRING_OPERATION": StringOperationActionConfig,
    "MATH_OPERATION": MathOperationActionConfig,
    # State actions
    "GO_TO_STATE": GoToStateActionConfig,
    "RUN_WORKFLOW": RunWorkflowActionConfig,
    "SCREENSHOT": ScreenshotActionConfig,
    # Code execution actions
    "CODE_BLOCK": CodeBlockActionConfig,
    "CUSTOM_FUNCTION": CustomFunctionActionConfig,
}


def get_typed_config(action: Action) -> BaseModel:
    """
    Get type-safe config model for an action.

    Args:
        action: The action to get config for

    Returns:
        Type-specific config model instance

    Raises:
        ValueError: If action type is unknown
        ValidationError: If config data is invalid
    """
    config_class = ACTION_CONFIG_MAP.get(action.type)
    if config_class is None:
        raise ValueError(f"Unknown action type: {action.type}")

    return config_class.model_validate(action.config)  # type: ignore[no-any-return, attr-defined]
