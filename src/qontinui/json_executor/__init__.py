"""JSON configuration executor for Qontinui automation.

This module provides the core execution engine for Qontinui automation workflows.
It supports both v2.0.0+ (Workflow-based) and v1.0.0 (Process-based) configurations
through backward-compatible aliases.
"""

from .action_executor import ActionExecutor
from .config_parser import ConfigParser
from .json_runner import JSONRunner
from .state_executor import StateExecutor

__all__ = ["ConfigParser", "StateExecutor", "ActionExecutor", "JSONRunner"]
