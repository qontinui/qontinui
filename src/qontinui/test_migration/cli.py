"""
Command-line interface for the Brobot test migration tool.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .config import TestMigrationConfig
    from .core.models import MigrationConfig
    from .orchestrator import TestMigrationOrchestrator
    from .reporting.dashboard import MigrationReportingDashboard
except ImportError:
    # Handle direct execution case
    from config import TestMigrationConfig
    from core.models import MigrationConfig
    from orchestrator import TestMigrationOrchestrator
    from reporting.dashboard import MigrationReportingDashboard


class TestMigrationCLI:
    """
    Command-line interface for the Brobot to Qontinui test migration tool.
    
    Provides commands for:
    - Migrating test suites
    - Validating migrations
    - Generating reports
    - Managing configuration
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
        self.dashboard = MigrationReportingDashboard()
    
    def run(self, args: Optional[List[str]] = None) -> int:
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
            elif parsed_args.command == "report":
                return self._handle_report_command(parsed_args)
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
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
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
            """
        )
        
        # Global options
        parser.add_argument(
            "-v", "--verbose",
            action="count",
            default=0,
            help="Increase verbosity (use -v, -vv, or -vvv)"
        )
        
        parser.add_argument(
            "--config",
            type=Path,
            help="Path to configuration file"
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Migrate command
        migrate_parser = subparsers.add_parser(
            "migrate",
            help="Migrate Java tests to Python"
        )
        self._add_migrate_arguments(migrate_parser)
        
        # Validate command
        validate_parser = subparsers.add_parser(
            "validate",
            help="Validate migrated tests"
        )
        self._add_validate_arguments(validate_parser)
        
        # Report command
        report_parser = subparsers.add_parser(
            "report",
            help="Generate migration reports"
        )
        self._add_report_arguments(report_parser)
        
        # Config command
        config_parser = subparsers.add_parser(
            "config",
            help="Manage configuration"
        )
        self._add_config_arguments(config_parser)
        
        return parser
    
    def _add_migrate_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the migrate command."""
        parser.add_argument(
            "source",
            type=Path,
            help="Source directory containing Java tests"
        )
        
        parser.add_argument(
            "target",
            type=Path,
            help="Target directory for Python tests"
        )
        
        parser.add_argument(
            "--preserve-structure",
            action="store_true",
            default=True,
            help="Preserve directory structure (default: True)"
        )
        
        parser.add_argument(
            "--no-preserve-structure",
            action="store_false",
            dest="preserve_structure",
            help="Don't preserve directory structure"
        )
        
        parser.add_argument(
            "--enable-mocks",
            action="store_true",
            default=True,
            help="Enable mock migration (default: True)"
        )
        
        parser.add_argument(
            "--no-mocks",
            action="store_false",
            dest="enable_mocks",
            help="Disable mock migration"
        )
        
        parser.add_argument(
            "--parallel",
            action="store_true",
            default=True,
            help="Enable parallel execution (default: True)"
        )
        
        parser.add_argument(
            "--no-parallel",
            action="store_false",
            dest="parallel",
            help="Disable parallel execution"
        )
        
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be migrated without actually doing it"
        )
        
        parser.add_argument(
            "--output-format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format for results (default: text)"
        )
        
        parser.add_argument(
            "--report-file",
            type=Path,
            help="Save migration report to file"
        )
    
    def _add_validate_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the validate command."""
        parser.add_argument(
            "test_directory",
            type=Path,
            help="Directory containing migrated Python tests"
        )
        
        parser.add_argument(
            "--compare-with",
            type=Path,
            help="Original Java test directory for comparison"
        )
        
        parser.add_argument(
            "--output-format",
            choices=["json", "yaml", "text"],
            default="text",
            help="Output format for results (default: text)"
        )
        
        parser.add_argument(
            "--report-file",
            type=Path,
            help="Save validation report to file"
        )
    
    def _add_report_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the report command."""
        parser.add_argument(
            "test_directory",
            type=Path,
            help="Directory containing migrated tests"
        )
        
        parser.add_argument(
            "--format",
            choices=["html", "json", "yaml", "text", "pdf"],
            default="html",
            help="Report format (default: html)"
        )
        
        parser.add_argument(
            "--output",
            type=Path,
            help="Output file for the report"
        )
        
        parser.add_argument(
            "--include-coverage",
            action="store_true",
            help="Include test coverage information"
        )
        
        parser.add_argument(
            "--include-diagnostics",
            action="store_true",
            help="Include diagnostic information"
        )
        
        parser.add_argument(
            "--template",
            type=Path,
            help="Custom report template file"
        )
    
    def _add_config_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments for the config command."""
        parser.add_argument(
            "--create",
            action="store_true",
            help="Create a new configuration file"
        )
        
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Validate an existing configuration file"
        )
        
        parser.add_argument(
            "--output",
            type=Path,
            help="Output file for configuration"
        )
        
        parser.add_argument(
            "--input",
            type=Path,
            help="Input configuration file to validate"
        )
    
    def _handle_migrate_command(self, args) -> int:
        """Handle the migrate command."""
        print(f"Migrating tests from {args.source} to {args.target}")
        
        # Validate source directory
        if not args.source.exists():
            print(f"Error: Source directory does not exist: {args.source}", file=sys.stderr)
            return 1
        
        # Create target directory if it doesn't exist
        args.target.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load or create configuration
            config = self._load_or_create_config(args)
            
            # Create orchestrator
            orchestrator = TestMigrationOrchestrator(config)
            
            if args.dry_run:
                return self._handle_dry_run(orchestrator, args.source, args.target)
            
            # Execute migration
            print("Starting migration process...")
            results = orchestrator.migrate_test_suite(args.source, args.target)
            
            # Display results
            self._display_migration_results(results, args.output_format)
            
            # Save report if requested
            if args.report_file:
                self._save_migration_report(results, args.report_file, orchestrator)
            
            # Return appropriate exit code
            return 0 if results.failed_tests == 0 else 1
            
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
            orchestrator = TestMigrationOrchestrator(config)
            
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
    
    def _handle_report_command(self, args) -> int:
        """Handle the report command."""
        print(f"Generating report for {args.test_directory}")
        
        if not args.test_directory.exists():
            print(f"Error: Test directory does not exist: {args.test_directory}", file=sys.stderr)
            return 1
        
        try:
            # Generate report
            report_data = self.dashboard.generate_comprehensive_report(
                args.test_directory,
                include_coverage=args.include_coverage,
                include_diagnostics=args.include_diagnostics
            )
            
            # Determine output file
            if args.output:
                output_file = args.output
            else:
                output_file = Path(f"migration_report.{args.format}")
            
            # Save report
            self.dashboard.save_report(
                report_data,
                output_file,
                format_type=args.format,
                template_file=args.template
            )
            
            print(f"Report saved to: {output_file}")
            return 0
            
        except Exception as e:
            print(f"Report generation failed: {str(e)}", file=sys.stderr)
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
                "exclude_patterns": config.exclude_patterns
            }
            
            # Determine output file
            output_file = args.output or Path("migration_config.json")
            
            # Save configuration
            with open(output_file, 'w') as f:
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
                "source_directories", "target_directory", "preserve_structure",
                "enable_mock_migration", "diagnostic_level", "parallel_execution"
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
        if hasattr(args, 'config') and args.config and args.config.exists():
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
            exclude_patterns=config_data.get("exclude_patterns", [])
        )
    
    def _create_config_from_args(self, args) -> MigrationConfig:
        """Create configuration from command line arguments."""
        source_dirs = [args.source] if hasattr(args, 'source') else []
        target_dir = args.target if hasattr(args, 'target') else Path("tests/migrated")
        
        return MigrationConfig(
            source_directories=source_dirs,
            target_directory=target_dir,
            preserve_structure=getattr(args, 'preserve_structure', True),
            enable_mock_migration=getattr(args, 'enable_mocks', True),
            diagnostic_level="detailed" if args.verbose > 1 else "normal" if args.verbose > 0 else "minimal",
            parallel_execution=getattr(args, 'parallel', True),
            comparison_mode="behavioral"
        )
    
    def _handle_dry_run(self, orchestrator: TestMigrationOrchestrator, source: Path, target: Path) -> int:
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
            
            return 0
            
        except Exception as e:
            print(f"Dry run failed: {str(e)}", file=sys.stderr)
            return 1
    
    def _display_migration_results(self, results, output_format: str):
        """Display migration results in the specified format."""
        if output_format == "json":
            self._display_results_json(results)
        elif output_format == "yaml":
            self._display_results_yaml(results)
        else:
            self._display_results_text(results)
    
    def _display_validation_results(self, results, output_format: str):
        """Display validation results in the specified format."""
        self._display_migration_results(results, output_format)
    
    def _display_results_text(self, results):
        """Display results in text format."""
        print("\nMigration Results:")
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
                    "error_message": r.error_message
                }
                for r in results.individual_results
            ]
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
                "execution_time": results.execution_time
            }
            print(yaml.dump(result_dict, default_flow_style=False))
        except ImportError:
            print("YAML output requires PyYAML package")
            self._display_results_text(results)
    
    def _save_migration_report(self, results, report_file: Path, orchestrator: TestMigrationOrchestrator):
        """Save migration report to file."""
        report_data = {
            "migration_results": results,
            "migration_state": orchestrator.migration_state,
            "configuration": orchestrator.config
        }
        
        self.dashboard.save_migration_report(report_data, report_file)
    
    def _save_validation_report(self, results, report_file: Path):
        """Save validation report to file."""
        report_data = {
            "validation_results": results,
            "timestamp": self.dashboard._get_timestamp()
        }
        
        with open(report_file, 'w') as f:
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
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def main():
    """Main entry point for the CLI."""
    cli = TestMigrationCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()