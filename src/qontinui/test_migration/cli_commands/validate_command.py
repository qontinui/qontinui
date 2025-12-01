"""Validate command handler."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from cli_commands.base_command import BaseCommand, CommandResult
from cli_utils.config_loader import ConfigLoader
from cli_utils.output_formatter import OutputFormatter
from minimal_orchestrator import MinimalMigrationOrchestrator


class ValidateCommand(BaseCommand):
    """
    Command to validate migrated Python tests.

    Validates previously migrated tests and reports results.
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add validate command arguments."""
        parser.add_argument(
            "test_directory",
            type=Path,
            help="Directory containing migrated Python tests",
        )

        parser.add_argument(
            "--output-format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format for results (default: text)",
        )

        parser.add_argument(
            "--report-file", type=Path, help="Save validation report to file"
        )

    def execute(self, args: Namespace) -> CommandResult:
        """Execute the validate command."""
        print(f"Validating migrated tests in {args.test_directory}")

        # Validate test directory exists
        if error := self.validate_path_exists(args.test_directory, "Test directory"):
            return error

        try:
            # Load configuration
            config = ConfigLoader.load_or_create(args)

            # Create orchestrator
            orchestrator = MinimalMigrationOrchestrator(config)

            # Execute validation
            print("Starting validation process...")
            results = orchestrator.validate_migration(args.test_directory)

            # Display results
            OutputFormatter.display_validation_results(results, args.output_format)

            # Save report if requested
            if args.report_file:
                OutputFormatter.save_validation_report(results, args.report_file)
                print(f"\nReport saved to: {args.report_file}")

            # Return error code if tests failed
            exit_code = 0 if results.failed_tests == 0 else 1
            return CommandResult(exit_code=exit_code)

        except Exception as e:
            return CommandResult(exit_code=1, message=f"Validation failed: {str(e)}")
