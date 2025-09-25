"""
Standalone command-line interface for the Brobot test migration tool.
This version handles import issues by using the minimal orchestrator.
"""

import argparse
import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import TestMigrationConfig  # noqa: E402
from core.models import MigrationConfig  # noqa: E402
from minimal_orchestrator import MinimalMigrationOrchestrator  # noqa: E402


class StandaloneTestMigrationCLI:
    """
    Standalone command-line interface for the Brobot to Qontinui test migration tool.

    This version uses the minimal orchestrator to avoid complex import dependencies.
    """

    def __init__(self):
        """Initialize the CLI."""
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
            if parsed_args.command == "migrate":
                return self._handle_migrate_command(parsed_args)
            elif parsed_args.command == "validate":
                return self._handle_validate_command(parsed_args)
            elif parsed_args.command == "discover":
                return self._handle_discover_command(parsed_args)
            elif parsed_args.command == "config":
                return self._handle_config_command(parsed_args)
            else:
                self.parser.print_help()
                return 1

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 130
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            if parsed_args.verbose > 2:
                import traceback

                traceback.print_exc()
            return 1

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
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

        # Discover command
        discover_parser = subparsers.add_parser(
            "discover", help="Discover Java tests in source directory"
        )
        self._add_discover_arguments(discover_parser)

        # Migrate command
        migrate_parser = subparsers.add_parser("migrate", help="Migrate Java tests to Python")
        self._add_migrate_arguments(migrate_parser)

        # Validate command
        validate_parser = subparsers.add_parser("validate", help="Validate migrated tests")
        self._add_validate_arguments(validate_parser)

        # Config command
        config_parser = subparsers.add_parser("config", help="Manage configuration")
        self._add_config_arguments(config_parser)

        return parser

    def _add_discover_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the discover command."""
        parser.add_argument("source", type=Path, help="Source directory containing Java tests")

        parser.add_argument(
            "--output-format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format for results (default: text)",
        )

        parser.add_argument("--output-file", type=Path, help="Save discovery results to file")

    def _add_migrate_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the migrate command."""
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
            "--no-mocks", action="store_false", dest="enable_mocks", help="Disable mock migration"
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

    def _add_validate_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the validate command."""
        parser.add_argument(
            "test_directory", type=Path, help="Directory containing migrated Python tests"
        )

        parser.add_argument(
            "--output-format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format for results (default: text)",
        )

        parser.add_argument("--report-file", type=Path, help="Save validation report to file")

    def _add_config_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the config command."""
        parser.add_argument("--create", action="store_true", help="Create a new configuration file")

        parser.add_argument(
            "--validate", action="store_true", help="Validate an existing configuration file"
        )

        parser.add_argument("--output", type=Path, help="Output file for configuration")

        parser.add_argument("--input", type=Path, help="Input configuration file to validate")

    def _handle_discover_command(self, args) -> int:
        """Handle the discover command."""
        print(f"Discovering tests in {args.source}")

        # Validate source directory
        if not args.source.exists():
            print(f"Error: Source directory does not exist: {args.source}", file=sys.stderr)
            return 1

        try:
            # Load or create configuration
            config = self._load_or_create_config(args)

            # Create orchestrator
            orchestrator = MinimalMigrationOrchestrator(config)

            # Discover tests
            print("Discovering tests...")
            discovered_tests = orchestrator.discover_tests(args.source)

            # Display results
            self._display_discovery_results(discovered_tests, args.output_format)

            # Save results if requested
            if args.output_file:
                self._save_discovery_results(discovered_tests, args.output_file)

            return 0

        except Exception as e:
            print(f"Discovery failed: {str(e)}", file=sys.stderr)
            return 1

    def _handle_migrate_command(self, args) -> int:
        """Handle the migrate command."""
        print(f"Migrating tests from {args.source} to {args.target}")

        # Validate source directory
        if not args.source.exists():
            print(f"Error: Source directory does not exist: {args.source}", file=sys.stderr)
            return 1

        # Create target directory if it doesn't exist
        if not args.dry_run:
            args.target.mkdir(parents=True, exist_ok=True)

        try:
            # Load or create configuration
            config = self._load_or_create_config(args)

            # Create orchestrator
            orchestrator = MinimalMigrationOrchestrator(config)

            if args.dry_run:
                return self._handle_dry_run(orchestrator, args.source, args.target)

            # For now, just do discovery since full migration has complex dependencies
            print("Starting test discovery...")
            discovered_tests = orchestrator.discover_tests(args.source)

            print(f"\nDiscovered {len(discovered_tests)} test files:")
            for i, test_file in enumerate(discovered_tests, 1):
                print(f"  {i}. {test_file.path.name} ({test_file.test_type.value})")

            print("\nNote: Full migration functionality requires additional components.")
            print("Use 'discover' command to analyze test files in detail.")

            return 0

        except Exception as e:
            print(f"Migration failed: {str(e)}", file=sys.stderr)
            return 1

    def _handle_validate_command(self, args) -> int:
        """Handle the validate command."""
        print(f"Validating migrated tests in {args.test_directory}")

        if not args.test_directory.exists():
            print(f"Error: Test directory does not exist: {args.test_directory}", file=sys.stderr)
            return 1

        try:
            # Load configuration
            config = self._load_or_create_config(args)

            # Create orchestrator
            orchestrator = MinimalMigrationOrchestrator(config)

            # Execute validation
            print("Starting validation process...")
            results = orchestrator.validate_migration(args.test_directory)

            # Display results
            self._display_validation_results(results, args.output_format)

            # Save report if requested
            if args.report_file:
                self._save_validation_report(results, args.report_file)

            return 0 if results.failed_tests == 0 else 1

        except Exception as e:
            print(f"Validation failed: {str(e)}", file=sys.stderr)
            return 1

    def _handle_config_command(self, args) -> int:
        """Handle the config command."""
        if args.create:
            return self._create_config_file(args)
        elif args.validate:
            return self._validate_config_file(args)
        else:
            print("Error: Must specify --create or --validate", file=sys.stderr)
            return 1

    def _create_config_file(self, args) -> int:
        """Create a new configuration file."""
        try:
            # Create default configuration
            config = TestMigrationConfig.create_default_config([], Path("tests/migrated"))

            # Convert to dictionary for serialization
            config_dict = {
                "source_directories": [str(d) for d in config.source_directories],
                "target_directory": str(config.target_directory),
                "preserve_structure": config.preserve_structure,
                "enable_mock_migration": config.enable_mock_migration,
                "diagnostic_level": config.diagnostic_level,
                "parallel_execution": config.parallel_execution,
                "comparison_mode": config.comparison_mode,
                "java_test_patterns": config.java_test_patterns,
                "exclude_patterns": config.exclude_patterns,
            }

            # Determine output file
            output_file = args.output or Path("migration_config.json")

            # Save configuration
            with open(output_file, "w") as f:
                json.dump(config_dict, f, indent=2)

            print(f"Configuration file created: {output_file}")
            return 0

        except Exception as e:
            print(f"Failed to create configuration: {str(e)}", file=sys.stderr)
            return 1

    def _validate_config_file(self, args) -> int:
        """Validate an existing configuration file."""
        config_file = args.input or args.config

        if not config_file or not config_file.exists():
            print("Error: Configuration file not found", file=sys.stderr)
            return 1

        try:
            # Load and validate configuration
            with open(config_file) as f:
                config_data = json.load(f)

            # Basic validation
            required_fields = [
                "source_directories",
                "target_directory",
                "preserve_structure",
                "enable_mock_migration",
                "diagnostic_level",
                "parallel_execution",
            ]

            missing_fields = [field for field in required_fields if field not in config_data]

            if missing_fields:
                print(f"Error: Missing required fields: {missing_fields}", file=sys.stderr)
                return 1

            print("Configuration file is valid")
            return 0

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error validating configuration: {str(e)}", file=sys.stderr)
            return 1

    def _load_or_create_config(self, args) -> MigrationConfig:
        """Load configuration from file or create from command line arguments."""
        if hasattr(args, "config") and args.config and args.config.exists():
            return self._load_config_from_file(args.config)
        else:
            return self._create_config_from_args(args)

    def _load_config_from_file(self, config_file: Path) -> MigrationConfig:
        """Load configuration from a JSON file."""
        with open(config_file) as f:
            config_data = json.load(f)

        return MigrationConfig(
            source_directories=[Path(d) for d in config_data["source_directories"]],
            target_directory=Path(config_data["target_directory"]),
            preserve_structure=config_data.get("preserve_structure", True),
            enable_mock_migration=config_data.get("enable_mock_migration", True),
            diagnostic_level=config_data.get("diagnostic_level", "detailed"),
            parallel_execution=config_data.get("parallel_execution", True),
            comparison_mode=config_data.get("comparison_mode", "behavioral"),
            java_test_patterns=config_data.get("java_test_patterns", ["*Test.java", "*Tests.java"]),
            exclude_patterns=config_data.get("exclude_patterns", []),
        )

    def _create_config_from_args(self, args) -> MigrationConfig:
        """Create configuration from command line arguments."""
        source_dirs = [args.source] if hasattr(args, "source") else []
        target_dir = args.target if hasattr(args, "target") else Path("tests/migrated")

        return MigrationConfig(
            source_directories=source_dirs,
            target_directory=target_dir,
            preserve_structure=getattr(args, "preserve_structure", True),
            enable_mock_migration=getattr(args, "enable_mocks", True),
            diagnostic_level=(
                "detailed" if args.verbose > 1 else "normal" if args.verbose > 0 else "minimal"
            ),
            parallel_execution=getattr(args, "parallel", True),
            comparison_mode="behavioral",
        )

    def _handle_dry_run(
        self, orchestrator: MinimalMigrationOrchestrator, source: Path, target: Path
    ) -> int:
        """Handle dry run mode."""
        print("DRY RUN MODE - No files will be modified")
        print("-" * 50)

        try:
            # Discover tests without migrating
            discovered_tests = orchestrator.discover_tests(source)

            print(f"Found {len(discovered_tests)} test files:")

            for test_file in discovered_tests:
                # Generate what the target path would be
                target_name = test_file.path.stem.replace("Test", "_test") + ".py"
                target_path = target / target_name

                print(f"  {test_file.path} -> {target_path}")
                print(f"    Type: {test_file.test_type.value}")
                print(f"    Package: {test_file.package}")
                if test_file.dependencies:
                    print(f"    Dependencies: {len(test_file.dependencies)}")
                    for dep in test_file.dependencies[:3]:  # Show first 3
                        print(f"      - {dep.java_import}")
                    if len(test_file.dependencies) > 3:
                        print(f"      ... and {len(test_file.dependencies) - 3} more")
                print()

            return 0

        except Exception as e:
            print(f"Dry run failed: {str(e)}", file=sys.stderr)
            return 1

    def _display_discovery_results(self, discovered_tests, output_format: str):
        """Display discovery results in the specified format."""
        if output_format == "json":
            self._display_discovery_json(discovered_tests)
        elif output_format == "yaml":
            self._display_discovery_yaml(discovered_tests)
        else:
            self._display_discovery_text(discovered_tests)

    def _display_discovery_text(self, discovered_tests):
        """Display discovery results in text format."""
        print(f"\nDiscovered {len(discovered_tests)} test files:")
        print("=" * 50)

        for i, test_file in enumerate(discovered_tests, 1):
            print(f"{i}. {test_file.path.name}")
            print(f"   Path: {test_file.path}")
            print(f"   Type: {test_file.test_type.value}")
            print(f"   Package: {test_file.package}")
            print(f"   Dependencies: {len(test_file.dependencies)}")

            if test_file.dependencies:
                print("   Key dependencies:")
                for dep in test_file.dependencies[:5]:  # Show first 5
                    print(f"     - {dep.java_import}")
                if len(test_file.dependencies) > 5:
                    print(f"     ... and {len(test_file.dependencies) - 5} more")
            print()

    def _display_discovery_json(self, discovered_tests):
        """Display discovery results in JSON format."""
        result_dict = {
            "total_files": len(discovered_tests),
            "test_files": [
                {
                    "name": test_file.path.name,
                    "path": str(test_file.path),
                    "type": test_file.test_type.value,
                    "package": test_file.package,
                    "dependencies": [dep.java_import for dep in test_file.dependencies],
                }
                for test_file in discovered_tests
            ],
        }
        print(json.dumps(result_dict, indent=2))

    def _display_discovery_yaml(self, discovered_tests):
        """Display discovery results in YAML format."""
        try:
            import yaml

            result_dict = {
                "total_files": len(discovered_tests),
                "test_files": [
                    {
                        "name": test_file.path.name,
                        "path": str(test_file.path),
                        "type": test_file.test_type.value,
                        "package": test_file.package,
                        "dependencies": [dep.java_import for dep in test_file.dependencies],
                    }
                    for test_file in discovered_tests
                ],
            }
            print(yaml.dump(result_dict, default_flow_style=False))
        except ImportError:
            print("YAML output requires PyYAML package")
            self._display_discovery_text(discovered_tests)

    def _display_validation_results(self, results, output_format: str):
        """Display validation results in the specified format."""
        if output_format == "json":
            self._display_results_json(results)
        elif output_format == "yaml":
            self._display_results_yaml(results)
        else:
            self._display_results_text(results)

    def _display_results_text(self, results):
        """Display results in text format."""
        print("\nValidation Results:")
        print("=" * 50)
        print(f"Total tests: {results.total_tests}")
        print(f"Passed: {results.passed_tests}")
        print(f"Failed: {results.failed_tests}")
        print(f"Skipped: {results.skipped_tests}")
        print(f"Execution time: {results.execution_time:.2f}s")

        if results.failed_tests > 0:
            print("\nFailed tests:")
            for result in results.individual_results:
                if not result.passed:
                    print(f"  - {result.test_name}: {result.error_message}")

    def _display_results_json(self, results):
        """Display results in JSON format."""
        result_dict = {
            "total_tests": results.total_tests,
            "passed_tests": results.passed_tests,
            "failed_tests": results.failed_tests,
            "skipped_tests": results.skipped_tests,
            "execution_time": results.execution_time,
            "individual_results": [
                {
                    "test_name": r.test_name,
                    "test_file": r.test_file,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                }
                for r in results.individual_results
            ],
        }
        print(json.dumps(result_dict, indent=2))

    def _display_results_yaml(self, results):
        """Display results in YAML format."""
        try:
            import yaml

            result_dict = {
                "total_tests": results.total_tests,
                "passed_tests": results.passed_tests,
                "failed_tests": results.failed_tests,
                "skipped_tests": results.skipped_tests,
                "execution_time": results.execution_time,
            }
            print(yaml.dump(result_dict, default_flow_style=False))
        except ImportError:
            print("YAML output requires PyYAML package")
            self._display_results_text(results)

    def _save_discovery_results(self, discovered_tests, output_file: Path):
        """Save discovery results to file."""
        result_dict = {
            "total_files": len(discovered_tests),
            "discovery_timestamp": str(Path(__file__).stat().st_mtime),
            "test_files": [
                {
                    "name": test_file.path.name,
                    "path": str(test_file.path),
                    "type": test_file.test_type.value,
                    "package": test_file.package,
                    "dependencies": [dep.java_import for dep in test_file.dependencies],
                }
                for test_file in discovered_tests
            ],
        }

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

    def _save_validation_report(self, results, report_file: Path):
        """Save validation report to file."""
        report_data = {
            "validation_results": {
                "total_tests": results.total_tests,
                "passed_tests": results.passed_tests,
                "failed_tests": results.failed_tests,
                "skipped_tests": results.skipped_tests,
                "execution_time": results.execution_time,
            },
            "individual_results": [
                {
                    "test_name": r.test_name,
                    "test_file": r.test_file,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                }
                for r in results.individual_results
            ],
        }

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def _configure_logging(self, verbose_level: int):
        """Configure logging based on verbosity level."""
        import logging

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
