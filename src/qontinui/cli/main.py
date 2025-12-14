"""Main CLI entry point for Qontinui automation."""

import sys
from pathlib import Path

import click

from .commands.run import run
from .commands.test import test
from .commands.validate import validate


@click.group(
    help="""Qontinui CLI - Run visual automation workflows from the command line.

Qontinui provides model-based GUI automation using computer vision and AI.
This CLI tool enables headless execution for CI/CD pipelines and testing.

Examples:

  # Run a specific workflow from a config file
  qontinui run config.json --workflow "Login Workflow"

  # Run in test mode with JUnit output
  qontinui test config.json --format junit --output ./test-results

  # Validate a config without running
  qontinui validate config.json

  # Stream results to qontinui-web
  qontinui test config.json --stream-to http://localhost:8000/api/v1/results
"""
)
@click.version_option(package_name="qontinui")
def main():
    """Qontinui CLI - Visual automation for CI/CD pipelines."""
    pass


# Register commands
main.add_command(run)
main.add_command(test)
main.add_command(validate)


if __name__ == "__main__":
    main()
