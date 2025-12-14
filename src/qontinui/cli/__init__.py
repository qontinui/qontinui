"""Command-line interface for Qontinui automation.

This module provides a comprehensive CLI for running Qontinui workflows in CI/CD
pipelines and headless environments. It supports multiple output formats, result
streaming, and detailed test reporting.
"""

from .main import main

__all__ = ["main"]
