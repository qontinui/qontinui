"""Utility functions for Qontinui CLI."""

import logging
import sys

import click


def configure_logging(verbose: bool):
    """Configure logging based on verbosity level.

    Args:
        verbose: If True, enable DEBUG logging; otherwise use INFO level
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def print_success(message: str):
    """Print a success message in green.

    Args:
        message: Message to print
    """
    click.echo(click.style(message, fg="green", bold=True))


def print_error(message: str):
    """Print an error message in red.

    Args:
        message: Error message to print
    """
    click.echo(click.style(f"Error: {message}", fg="red", bold=True), err=True)


def print_warning(message: str):
    """Print a warning message in yellow.

    Args:
        message: Warning message to print
    """
    click.echo(click.style(f"Warning: {message}", fg="yellow", bold=True))


def print_info(message: str):
    """Print an info message in blue.

    Args:
        message: Info message to print
    """
    click.echo(click.style(message, fg="blue"))
