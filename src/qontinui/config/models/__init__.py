"""
Qontinui action and workflow configuration models.

This package provides Pydantic models for type-safe action and workflow
configuration, organized by domain for maintainability.
"""

# Base types and enums
# Core action model
from .action import ACTION_CONFIG_MAP, Action, get_typed_config
from .base_types import (
    LogLevel,
    MouseButton,
    SearchStrategy,
    VerificationMode,
    WorkflowVisibility,
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

# Execution control
from .execution import BaseActionSettings, ExecutionSettings, RepetitionOptions

# Find action configs
from .find_actions import (
    FindActionConfig,
    FindStateImageActionConfig,
    VanishActionConfig,
    WaitActionConfig,
    WaitCondition,
)

# Geometry primitives
from .geometry import Coordinates, Region

# Keyboard action configs
from .keyboard_actions import (
    HotkeyActionConfig,
    KeyDownActionConfig,
    KeyPressActionConfig,
    KeyUpActionConfig,
    TextSource,
    TypeActionConfig,
)

# Logging configuration
from .logging import LoggingOptions

# Mouse action configs
from .mouse_actions import (
    ClickActionConfig,
    DragActionConfig,
    HighlightActionConfig,
    MouseDownActionConfig,
    MouseMoveActionConfig,
    MouseUpActionConfig,
    ScrollActionConfig,
)

# Search and pattern matching
from .search import (
    MatchAdjustment,
    PatternOptions,
    PollingConfig,
    SearchOptions,
    TextSearchOptions,
)

# State and workflow action configs
from .state_actions import (
    GoToStateActionConfig,
    RunWorkflowActionConfig,
    ScreenshotActionConfig,
    ScreenshotSaveConfig,
    WorkflowRepetition,
)

# Target configurations
from .targets import (
    CoordinatesTarget,
    CurrentPositionTarget,
    ImageTarget,
    LastFindResultTarget,
    RegionTarget,
    StateStringTarget,
    TargetConfig,
    TextTarget,
)

# Verification
from .verification import VerificationConfig

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
    "LastFindResultTarget",
    "RegionTarget",
    "StateStringTarget",
    "TargetConfig",
    "TextTarget",
    # Verification
    "VerificationConfig",
    # Mouse actions
    "ClickActionConfig",
    "DragActionConfig",
    "HighlightActionConfig",
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
