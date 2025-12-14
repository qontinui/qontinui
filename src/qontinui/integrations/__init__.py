"""Integrations with external tools and services.

This module provides integrations with external tools like Claude Code CLI.
"""

from .claude import (
    find_claude,
    get_configured_claude_path,
    is_claude_available,
    run_claude,
    set_configured_claude_path,
    to_wsl_path,
)

__all__ = [
    "find_claude",
    "get_configured_claude_path",
    "is_claude_available",
    "run_claude",
    "set_configured_claude_path",
    "to_wsl_path",
]
