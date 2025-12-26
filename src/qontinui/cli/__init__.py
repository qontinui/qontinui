"""Qontinui Command Line Interface.

Provides CLI commands for:
- Running workflows
- Running integration tests in mock mode
- Validating workflow configurations
- Converting between configuration formats

Usage:
    python -m qontinui.cli --help
    python -m qontinui.cli run --config workflow.json
    python -m qontinui.cli integration-test --workflow workflow.json --mock

Or via the installed entry point:
    qontinui --help
"""

from .main import main

__all__ = ["main"]
