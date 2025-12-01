"""
Command-line interface for the Brobot test migration tool.

Refactored to use Command Pattern for better separation of concerns.
All command logic is delegated to specialized command handlers.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

from cli_commands import (
    BaseCommand,
    ConfigCommand,
    MigrateCommand,
    ReportCommand,
    ValidateCommand,
)


class TestMigrationCLI:
    """
    Command-line interface for the Brobot to Qontinui test migration tool.

    Uses Command Pattern to delegate to specialized command handlers:
    - MigrateCommand: Migration workflow
    - ValidateCommand: Validation workflow
    - ReportCommand: Report generation
    - ConfigCommand: Configuration management
    """

    def __init__(self) -> None:
        """Initialize the CLI with command handlers."""
        self.parser = self._create_parser()
        self.commands: dict[str, BaseCommand] = {
            "migrate": MigrateCommand(),
            "validate": ValidateCommand(),
            "report": ReportCommand(),
            "config": ConfigCommand(),
        }

    def run(self, args: list[str] | None = None) -> int:
        """
        Run the CLI with the given arguments.

        Args:
            args: Command line arguments (uses sys.argv if None)

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            parsed_args = self.parser.parse_args(args)

            # Set up logging level based on verbosity
            self._configure_logging(parsed_args.verbose)

            # Execute the appropriate command
            if parsed_args.command in self.commands:
                command = self.commands[parsed_args.command]
                result = command.execute(parsed_args)

                if result.message:
                    print(result.message)

                return cast(int, result.exit_code)
            else:
                self.parser.print_help()
                return 1

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 130
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return 1

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            prog="qontinui-test-migration",
            description="Migrate Brobot Java tests to Qontinui Python tests",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Migrate tests from Brobot to Qontinui
  qontinui-test-migration migrate /path/to/brobot/tests /path/to/qontinui/tests

  # Validate previously migrated tests
  qontinui-test-migration validate /path/to/qontinui/tests

  # Generate migration report
  qontinui-test-migration report /path/to/qontinui/tests --format html

  # Create configuration file
  qontinui-test-migration config --create --output migration.json
            """,
        )

        # Global options
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Increase verbosity (use -v, -vv, or -vvv)",
        )

        parser.add_argument("--config", type=Path, help="Path to configuration file")

        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Add each command's arguments
        migrate_parser = subparsers.add_parser("migrate", help="Migrate Java tests to Python")
        self.commands["migrate"].add_arguments(migrate_parser)

        validate_parser = subparsers.add_parser("validate", help="Validate migrated tests")
        self.commands["validate"].add_arguments(validate_parser)

        report_parser = subparsers.add_parser("report", help="Generate migration reports")
        self.commands["report"].add_arguments(report_parser)

        config_parser = subparsers.add_parser("config", help="Manage configuration")
        self.commands["config"].add_arguments(config_parser)

        return parser

    def _configure_logging(self, verbose_level: int):
        """Configure logging based on verbosity level."""
        if verbose_level >= 3:
            level = logging.DEBUG
        elif verbose_level >= 2:
            level = logging.INFO
        elif verbose_level >= 1:
            level = logging.WARNING
        else:
            level = logging.ERROR

        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def main():
    """Main entry point for the CLI."""
    cli = TestMigrationCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
