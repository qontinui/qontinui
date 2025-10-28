"""DEPRECATED: This file has been replaced by the modular action executor system.

The monolithic ActionExecutor class has been refactored into a modular system
with specialized executors for each action type group. This provides better
separation of concerns, easier testing, and clearer code organization.

Migration Guide
===============

Old import:
    from qontinui.json_executor.action_executor import ActionExecutor

New import:
    from qontinui.action_executors import DelegatingActionExecutor

The API is fully compatible - DelegatingActionExecutor has the same interface
as the old ActionExecutor class.

Architecture Changes
====================

Old: Monolithic ActionExecutor (2000+ lines)
New: Modular system with:
    - DelegatingActionExecutor: Main executor that delegates to specialized executors
    - MouseActionExecutor: Handles CLICK, DRAG, MOUSE_MOVE, etc.
    - KeyboardActionExecutor: Handles TYPE, KEY_PRESS, KEY_DOWN, etc.
    - VisionActionExecutor: Handles FIND, EXISTS, VANISH
    - NavigationActionExecutor: Handles GO_TO_STATE, RUN_WORKFLOW
    - UtilityActionExecutor: Handles WAIT, SCREENSHOT
    - ControlFlowActionExecutor: Handles LOOP, IF, BREAK, CONTINUE
    - DataOperationsActionExecutor: Handles SET_VARIABLE, MAP, FILTER, etc.

See: src/qontinui/action_executors/ for the new implementation

History
=======
- 2025-01-27: Replaced with redirect to new modular system
- 2025-01-26: Last version of monolithic implementation
"""

# Redirect to new implementation
from ..action_executors import DelegatingActionExecutor

__all__ = ["DelegatingActionExecutor"]
