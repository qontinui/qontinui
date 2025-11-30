"""
Standalone command-line interface for the Brobot test migration tool.

This version uses the Command Pattern for clean separation of concerns.
Each command (migrate, discover, validate, config) has its own handler module.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli_commands import (  # noqa: E402
    BaseCommand,
    ConfigCommand,
    DiscoverCommand,
    MigrateCommand,
    ValidateCommand,
)


class StandaloneTestMigrationCLI:
    """
    Standalone command-line interface for the Brobot to Qontinui test migration tool.

    Implements the Command Pattern with dedicated command handlers for each operation.
    """

    def __init__(self) -> None:
        """Initialize the CLI with command handlers."""
        self.commands: dict[str, BaseCommand] = {
            "discover": DiscoverCommand(),
            "migrate": MigrateCommand(),
            "validate": ValidateCommand(),
            "config": ConfigCommand(),
        }
        self.parser = self._create_parser()

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
                    if result.exit_code == 0:
                        print(result.message)
                    else:
                        logging.error(result.message)

                return cast(int, result.exit_code)
            else:
                self.parser.print_help()
                return 1

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 130
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            if hasattr(parsed_args, "verbose") and parsed_args.verbose > 2:
                import traceback

                traceback.print_exc()
            return 1

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            prog="qontinui-test-migration",
            description="Migrate Brobot Java tests to Qontinui Python tests",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Discover tests in Brobot directory
  python cli_standalone.py discover /path/to/brobot/tests

  # Preview migration (dry run)
  python cli_standalone.py migrate /path/to/brobot/tests /path/to/qontinui/tests --dry-run

  # Migrate tests from Brobot to Qontinui
  python cli_standalone.py migrate /path/to/brobot/tests /path/to/qontinui/tests

  # Validate previously migrated tests
  python cli_standalone.py validate /path/to/qontinui/tests

  # Create configuration file
  python cli_standalone.py config --create --output migration.json
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
        for command_name, command_handler in self.commands.items():
            subparser = subparsers.add_parser(
                command_name, help=f"{command_name.capitalize()} command"
            )
            command_handler.add_arguments(subparser)

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
    cli = StandaloneTestMigrationCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
