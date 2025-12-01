"""Migrate command handler."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from cli_commands.base_command import BaseCommand, CommandResult
from cli_utils.config_loader import ConfigLoader
from cli_utils.output_formatter import OutputFormatter
from orchestrator import TestMigrationOrchestrator


class MigrateCommand(BaseCommand):
    """
    Command to migrate Java tests to Python.

    Handles both dry-run mode (preview) and actual migration.
    Uses TestMigrationOrchestrator for full migration workflow.
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add migrate command arguments."""
        parser.add_argument("source", type=Path, help="Source directory containing Java tests")

        parser.add_argument("target", type=Path, help="Target directory for Python tests")

        parser.add_argument(
            "--preserve-structure",
            action="store_true",
            default=True,
            help="Preserve directory structure (default: True)",
        )

        parser.add_argument(
            "--no-preserve-structure",
            action="store_false",
            dest="preserve_structure",
            help="Don't preserve directory structure",
        )

        parser.add_argument(
            "--enable-mocks",
            action="store_true",
            default=True,
            help="Enable mock migration (default: True)",
        )

        parser.add_argument(
            "--no-mocks",
            action="store_false",
            dest="enable_mocks",
            help="Disable mock migration",
        )

        parser.add_argument(
            "--parallel",
            action="store_true",
            default=True,
            help="Enable parallel execution (default: True)",
        )

        parser.add_argument(
            "--no-parallel",
            action="store_false",
            dest="parallel",
            help="Disable parallel execution",
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be migrated without actually doing it",
        )

        parser.add_argument(
            "--output-format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format for results (default: text)",
        )

        parser.add_argument("--report-file", type=Path, help="Save migration report to file")

    def execute(self, args: Namespace) -> CommandResult:
        """Execute the migrate command."""
        print(f"Migrating tests from {args.source} to {args.target}")

        # Validate source directory
        if error := self.validate_path_exists(args.source, "Source directory"):
            return error

        # Create target directory if it doesn't exist
        args.target.mkdir(parents=True, exist_ok=True)

        try:
            # Load or create configuration
            config = ConfigLoader.load_or_create(args)

            # Create orchestrator
            orchestrator = TestMigrationOrchestrator(config)

            if args.dry_run:
                return self._handle_dry_run(orchestrator, args.source, args.target)

            # Execute migration
            print("Starting migration process...")
            results = orchestrator.migrate_test_suite(args.source, args.target)

            # Display results
            OutputFormatter.display_migration_results(results, args.output_format)

            # Save report if requested
            if args.report_file:
                OutputFormatter.save_migration_report(results, args.report_file, orchestrator)

            # Return appropriate exit code
            return CommandResult(exit_code=0 if results.failed_tests == 0 else 1)

        except Exception as e:
            return CommandResult(exit_code=1, message=f"Migration failed: {str(e)}")

    def _handle_dry_run(
        self, orchestrator: TestMigrationOrchestrator, source: Path, target: Path
    ) -> CommandResult:
        """Handle dry run mode."""
        print("DRY RUN MODE - No files will be modified")
        print("-" * 50)

        try:
            # Discover tests without migrating
            discovered_tests = orchestrator._discover_tests(source)

            print(f"Found {len(discovered_tests)} test files:")

            for test_file in discovered_tests:
                target_path = orchestrator._generate_target_path(test_file, target)
                print(f"  {test_file.path} -> {target_path}")
                print(f"    Type: {test_file.test_type.value}")
                print(f"    Package: {test_file.package}")
                if test_file.mock_usage:
                    print(f"    Mock usage: {len(test_file.mock_usage)} mocks")
                print()

            return CommandResult(exit_code=0)

        except Exception as e:
            return CommandResult(exit_code=1, message=f"Dry run failed: {str(e)}")
