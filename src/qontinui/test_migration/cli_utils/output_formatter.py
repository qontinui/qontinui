"""Output formatting utilities for CLI commands."""

import json
from pathlib import Path
from typing import Any


class OutputFormatter:
    """
    Handles formatting and display of command results.

    Supports multiple output formats:
    - text: Human-readable text output
    - json: JSON formatted output
    - yaml: YAML formatted output (requires PyYAML)
    """

    @staticmethod
    def display_discovery_results(discovered_tests, output_format: str) -> None:
        """Display discovery results in the specified format."""
        if output_format == "json":
            OutputFormatter._display_discovery_json(discovered_tests)
        elif output_format == "yaml":
            OutputFormatter._display_discovery_yaml(discovered_tests)
        else:
            OutputFormatter._display_discovery_text(discovered_tests)

    @staticmethod
    def _display_discovery_text(discovered_tests) -> None:
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

    @staticmethod
    def _display_discovery_json(discovered_tests) -> None:
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

    @staticmethod
    def _display_discovery_yaml(discovered_tests) -> None:
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
            OutputFormatter._display_discovery_text(discovered_tests)

    @staticmethod
    def display_validation_results(results, output_format: str) -> None:
        """Display validation results in the specified format."""
        if output_format == "json":
            OutputFormatter._display_validation_json(results)
        elif output_format == "yaml":
            OutputFormatter._display_validation_yaml(results)
        else:
            OutputFormatter._display_validation_text(results)

    @staticmethod
    def _display_validation_text(results) -> None:
        """Display validation results in text format."""
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

    @staticmethod
    def _display_validation_json(results) -> None:
        """Display validation results in JSON format."""
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

    @staticmethod
    def _display_validation_yaml(results) -> None:
        """Display validation results in YAML format."""
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
            OutputFormatter._display_validation_text(results)

    @staticmethod
    def save_discovery_results(discovered_tests, output_file: Path) -> None:
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

    @staticmethod
    def save_validation_report(results, report_file: Path) -> None:
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

    @staticmethod
    def display_migration_results(results, output_format: str) -> None:
        """Display migration results in the specified format."""
        if output_format == "json":
            OutputFormatter._display_migration_json(results)
        elif output_format == "yaml":
            OutputFormatter._display_migration_yaml(results)
        else:
            OutputFormatter._display_migration_text(results)

    @staticmethod
    def _display_migration_text(results) -> None:
        """Display migration results in text format."""
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

    @staticmethod
    def _display_migration_json(results) -> None:
        """Display migration results in JSON format."""
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

    @staticmethod
    def _display_migration_yaml(results) -> None:
        """Display migration results in YAML format."""
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
            OutputFormatter._display_migration_text(results)

    @staticmethod
    def save_migration_report(results, report_file: Path, orchestrator) -> None:
        """Save migration report to file."""
        report_data = {
            "migration_results": {
                "total_tests": results.total_tests,
                "passed_tests": results.passed_tests,
                "failed_tests": results.failed_tests,
                "skipped_tests": results.skipped_tests,
                "execution_time": results.execution_time,
            },
            "migration_state": str(orchestrator.migration_state),
            "configuration": str(orchestrator.config),
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
