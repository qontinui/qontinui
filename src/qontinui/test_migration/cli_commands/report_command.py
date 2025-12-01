"""Report command handler."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from cli_commands.base_command import BaseCommand, CommandResult
from reporting.dashboard import MigrationReportingDashboard


class ReportCommand(BaseCommand):
    """
    Command to generate migration reports.

    Supports multiple report formats including HTML, JSON, YAML, text, and PDF.
    Can include coverage and diagnostic information.
    """

    def __init__(self) -> None:
        """Initialize the report command."""
        self.dashboard = MigrationReportingDashboard()

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add report command arguments."""
        parser.add_argument(
            "test_directory", type=Path, help="Directory containing migrated tests"
        )

        parser.add_argument(
            "--format",
            choices=["html", "json", "yaml", "text", "pdf"],
            default="html",
            help="Report format (default: html)",
        )

        parser.add_argument("--output", type=Path, help="Output file for the report")

        parser.add_argument(
            "--include-coverage",
            action="store_true",
            help="Include test coverage information",
        )

        parser.add_argument(
            "--include-diagnostics",
            action="store_true",
            help="Include diagnostic information",
        )

        parser.add_argument("--template", type=Path, help="Custom report template file")

    def execute(self, args: Namespace) -> CommandResult:
        """Execute the report command."""
        print(f"Generating report for {args.test_directory}")

        # Validate test directory
        if error := self.validate_path_exists(args.test_directory, "Test directory"):
            return error

        try:
            # Generate report
            report_data = self.dashboard.generate_comprehensive_report(
                args.test_directory,
                include_coverage=args.include_coverage,
                include_diagnostics=args.include_diagnostics,
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
                template_file=args.template,
            )

            return CommandResult(exit_code=0, message=f"Report saved to: {output_file}")

        except Exception as e:
            return CommandResult(
                exit_code=1, message=f"Report generation failed: {str(e)}"
            )
