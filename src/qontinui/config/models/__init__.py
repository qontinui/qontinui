"""
Qontinui action and workflow configuration models.

This package provides Pydantic models for type-safe action and workflow
configuration, organized by domain for maintainability.
"""

# Base types and enums
from .base_types import (
    LogLevel,
    MouseButton,
    SearchStrategy,
    VerificationMode,
    WorkflowVisibility,
)

# Geometry primitives
from .geometry import Coordinates, Region

# Logging configuration
from .logging import LoggingOptions

# Execution control
from .execution import BaseActionSettings, ExecutionSettings, RepetitionOptions

# Search and pattern matching
from .search import (
    MatchAdjustment,
    PatternOptions,
    PollingConfig,
    SearchOptions,
    TextSearchOptions,
)

# Target configurations
from .targets import (
    CoordinatesTarget,
    CurrentPositionTarget,
    ImageTarget,
    RegionTarget,
    StateStringTarget,
    TargetConfig,
    TextTarget,
)

# Verification
from .verification import VerificationConfig

# Mouse action configs
from .mouse_actions import (
    ClickActionConfig,
    DragActionConfig,
    MouseDownActionConfig,
    MouseMoveActionConfig,
    MouseUpActionConfig,
    ScrollActionConfig,
)

# Keyboard action configs
from .keyboard_actions import (
    HotkeyActionConfig,
    KeyDownActionConfig,
    KeyPressActionConfig,
    KeyUpActionConfig,
    TextSource,
    TypeActionConfig,
)

# Find action configs
from .find_actions import (
    ExistsActionConfig,
    FindActionConfig,
    FindStateImageActionConfig,
    VanishActionConfig,
    WaitActionConfig,
    WaitCondition,
)

# Control flow configs
from .control_flow import (
    BreakActionConfig,
    ConditionConfig,
    ContinueActionConfig,
    IfActionConfig,
    LoopActionConfig,
    LoopCollection,
    SwitchActionConfig,
    SwitchCase,
    TryCatchActionConfig,
)

# Data operation configs
from .data_operations import (
    FilterActionConfig,
    FilterCondition,
    GetVariableActionConfig,
    MapActionConfig,
    MapTransform,
    MathOperationActionConfig,
    ReduceActionConfig,
    SetVariableActionConfig,
    SortActionConfig,
    StringOperationActionConfig,
    StringOperationParameters,
    ValueSource,
)

# State and workflow action configs
from .state_actions import (
    GoToStateActionConfig,
    RunWorkflowActionConfig,
    ScreenshotActionConfig,
    ScreenshotSaveConfig,
    WorkflowRepetition,
)

# Core action model
from .action import ACTION_CONFIG_MAP, Action, get_typed_config

# Workflow models
from .workflow import (
    Connection,
    Connections,
    Variables,
    Workflow,
    WorkflowMetadata,
    WorkflowSettings,
)

__all__ = [
    # Base types
    "LogLevel",
    "MouseButton",
    "SearchStrategy",
    "VerificationMode",
    "WorkflowVisibility",
    # Geometry
    "Coordinates",
    "Region",
    # Logging
    "LoggingOptions",
    # Execution
    "BaseActionSettings",
    "ExecutionSettings",
    "RepetitionOptions",
    # Search
    "MatchAdjustment",
    "PatternOptions",
    "PollingConfig",
    "SearchOptions",
    "TextSearchOptions",
    # Targets
    "CoordinatesTarget",
    "CurrentPositionTarget",
    "ImageTarget",
    "RegionTarget",
    "StateStringTarget",
    "TargetConfig",
    "TextTarget",
    # Verification
    "VerificationConfig",
    # Mouse actions
    "ClickActionConfig",
    "DragActionConfig",
    "MouseDownActionConfig",
    "MouseMoveActionConfig",
    "MouseUpActionConfig",
    "ScrollActionConfig",
    # Keyboard actions
    "HotkeyActionConfig",
    "KeyDownActionConfig",
    "KeyPressActionConfig",
    "KeyUpActionConfig",
    "TextSource",
    "TypeActionConfig",
    # Find actions
    "ExistsActionConfig",
    "FindActionConfig",
    "FindStateImageActionConfig",
    "VanishActionConfig",
    "WaitActionConfig",
    "WaitCondition",
    # Control flow
    "BreakActionConfig",
    "ConditionConfig",
    "ContinueActionConfig",
    "IfActionConfig",
    "LoopActionConfig",
    "LoopCollection",
    "SwitchActionConfig",
    "SwitchCase",
    "TryCatchActionConfig",
    # Data operations
    "FilterActionConfig",
    "FilterCondition",
    "GetVariableActionConfig",
    "MapActionConfig",
    "MapTransform",
    "MathOperationActionConfig",
    "ReduceActionConfig",
    "SetVariableActionConfig",
    "SortActionConfig",
    "StringOperationActionConfig",
    "StringOperationParameters",
    "ValueSource",
    # State actions
    "GoToStateActionConfig",
    "RunWorkflowActionConfig",
    "ScreenshotActionConfig",
    "ScreenshotSaveConfig",
    "WorkflowRepetition",
    # Core action
    "ACTION_CONFIG_MAP",
    "Action",
    "get_typed_config",
    # Workflow
    "Connection",
    "Connections",
    "Variables",
    "Workflow",
    "WorkflowMetadata",
    "WorkflowSettings",
]
