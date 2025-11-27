"""Base command interface following Command Pattern."""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    """Result of command execution."""

    exit_code: int
    message: str | None = None


class BaseCommand(ABC):
    """
    Abstract base class for CLI commands.

    Implements the Command Pattern for CLI command handlers.
    Each command is responsible for:
    - Defining its own arguments
    - Executing its specific logic
    - Returning a CommandResult
    """

    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """
        Add command-specific arguments to the parser.

        Args:
            parser: ArgumentParser for this command
        """
        pass

    @abstractmethod
    def execute(self, args: Namespace) -> CommandResult:
        """
        Execute the command with the given arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            CommandResult with exit code and optional message
        """
        pass

    def validate_path_exists(self, path: Path, path_name: str) -> CommandResult | None:
        """
        Validate that a path exists.

        Args:
            path: Path to validate
            path_name: Name to use in error message

        Returns:
            CommandResult with error if path doesn't exist, None otherwise
        """
        if not path.exists():
            return CommandResult(exit_code=1, message=f"Error: {path_name} does not exist: {path}")
        return None
