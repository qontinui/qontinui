"""Discover command handler."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from cli_commands.base_command import BaseCommand, CommandResult
from cli_utils.config_loader import ConfigLoader
from cli_utils.output_formatter import OutputFormatter
from minimal_orchestrator import MinimalMigrationOrchestrator


class DiscoverCommand(BaseCommand):
    """
    Command to discover Java tests in source directory.

    Scans the source directory for Java test files and reports
    findings in various formats (text, JSON, YAML).
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add discover command arguments."""
        parser.add_argument(
            "source", type=Path, help="Source directory containing Java tests"
        )

        parser.add_argument(
            "--output-format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format for results (default: text)",
        )

        parser.add_argument(
            "--output-file", type=Path, help="Save discovery results to file"
        )

    def execute(self, args: Namespace) -> CommandResult:
        """Execute the discover command."""
        print(f"Discovering tests in {args.source}")

        # Validate source directory
        if error := self.validate_path_exists(args.source, "Source directory"):
            return error

        try:
            # Load or create configuration
            config = ConfigLoader.load_or_create(args)

            # Create orchestrator
            orchestrator = MinimalMigrationOrchestrator(config)

            # Discover tests
            print("Discovering tests...")
            discovered_tests = orchestrator.discover_tests(args.source)

            # Display results
            OutputFormatter.display_discovery_results(
                discovered_tests, args.output_format
            )

            # Save results if requested
            if args.output_file:
                OutputFormatter.save_discovery_results(
                    discovered_tests, args.output_file
                )
                print(f"\nResults saved to: {args.output_file}")

            return CommandResult(exit_code=0)

        except Exception as e:
            return CommandResult(exit_code=1, message=f"Discovery failed: {str(e)}")
