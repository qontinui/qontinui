"""Debugging subsystem for qontinui.

This package provides comprehensive debugging capabilities for the qontinui
automation framework including:

- Debug sessions with pause/step/continue control
- Breakpoints (action ID, type, conditional, error)
- Execution recording and history
- Variable snapshots
- Performance tracking
- Integration hooks

Example:
    >>> from qontinui.debugging import DebugManager
    >>> debug = DebugManager.get_instance()
    >>> debug.enable_debugging()
    >>> session = debug.create_session("my_test")
    >>> debug.breakpoints.add_type_breakpoint("Click")
"""

from .breakpoint_manager import BreakpointManager
from .debug_manager import DebugManager
from .debug_session import DebugSession
from .execution_recorder import ExecutionRecorder
from .types import (
    Breakpoint,
    BreakpointType,
    DebugHookContext,
    ErrorHook,
    ExecutionRecord,
    ExecutionState,
    PostActionHook,
    PreActionHook,
    StepMode,
    VariableSnapshot,
)

__all__ = [
    # Main classes
    "DebugManager",
    "DebugSession",
    "BreakpointManager",
    "ExecutionRecorder",
    # Types and enums
    "BreakpointType",
    "ExecutionState",
    "StepMode",
    "Breakpoint",
    "ExecutionRecord",
    "VariableSnapshot",
    "DebugHookContext",
    # Hook types
    "PreActionHook",
    "PostActionHook",
    "ErrorHook",
]
