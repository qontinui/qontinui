"""
Migration reporting dashboard for generating comprehensive reports.
"""

import datetime
import json
from pathlib import Path
from typing import Any

try:
    from ..validation.coverage_tracker import CoverageTracker
except ImportError:
    # Handle direct execution case
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from validation.coverage_tracker import CoverageTracker


class MigrationReportingDashboard:
    """
    Comprehensive reporting dashboard for test migration results.

    Generates reports in multiple formats:
    - HTML dashboard with interactive elements
    - JSON data for programmatic access
    - Text summaries for console output
    - PDF reports for documentation
    """

    def __init__(self, java_source_dir: Path = None, python_target_dir: Path = None):
        """Initialize the reporting dashboard."""
        # Use default paths if not provided
        if java_source_dir is None:
            java_source_dir = Path("./java_tests")
        if python_target_dir is None:
            python_target_dir = Path("./python_tests")

        self.coverage_tracker = CoverageTracker(java_source_dir, python_target_dir)
        from validation.diagnostic_reporter import DiagnosticReporterImpl

        self.diagnostic_reporter = DiagnosticReporterImpl()

    def generate_comprehensive_report(
        self, test_directory: Path, include_coverage: bool = True, include_diagnostics: bool = True
    ) -> dict[str, Any]:
        """
        Generate a comprehensive migration report.

        Args:
            test_directory: Directory containing migrated tests
            include_coverage: Whether to include coverage information
            include_diagnostics: Whether to include diagnostic information

        Returns:
            Dictionary containing all report data
        """
        report_data = {
            "metadata": self._generate_metadata(),
            "summary": self._generate_summary(test_directory),
            "test_results": self._analyze_test_results(test_directory),
            "migration_statistics": self._generate_migration_statistics(test_directory),
        }

        if include_coverage:
            report_data["coverage"] = self._generate_coverage_report(test_directory)

        if include_diagnostics:
            report_data["diagnostics"] = self._generate_diagnostic_report(test_directory)

        return report_data

    def save_report(
        self,
        report_data: dict[str, Any],
        output_file: Path,
        format_type: str = "html",
        template_file: Path | None = None,
    ):
        """
        Save report in the specified format.

        Args:
            report_data: Report data dictionary
            output_file: Output file path
            format_type: Format type (html, json, yaml, text, pdf)
            template_file: Optional custom template file
        """
        if format_type == "html":
            self._save_html_report(report_data, output_file, template_file)
        elif format_type == "json":
            self._save_json_report(report_data, output_file)
        elif format_type == "yaml":
            self._save_yaml_report(report_data, output_file)
        elif format_type == "text":
            self._save_text_report(report_data, output_file)
        elif format_type == "pdf":
            self._save_pdf_report(report_data, output_file)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def save_migration_report(self, report_data: dict[str, Any], output_file: Path):
        """Save a migration-specific report."""
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def _generate_metadata(self) -> dict[str, Any]:
        """Generate report metadata."""
        return {
            "generated_at": self._get_timestamp(),
            "generator": "Qontinui Test Migration Tool",
            "version": "1.0.0",
        }

    def _generate_summary(self, test_directory: Path) -> dict[str, Any]:
        """Generate high-level summary."""
        # Count test files
        test_files = list(test_directory.rglob("test_*.py")) + list(
            test_directory.rglob("*_test.py")
        )

        return {
            "test_directory": str(test_directory),
            "total_test_files": len(test_files),
            "test_file_list": [str(f.relative_to(test_directory)) for f in test_files],
        }

    def _analyze_test_results(self, test_directory: Path) -> dict[str, Any]:
        """Analyze test execution results."""
        # This would typically load results from a previous test run
        # For now, return placeholder data
        return {
            "execution_status": "not_executed",
            "message": "Run tests to populate execution results",
        }

    def _generate_migration_statistics(self, test_directory: Path) -> dict[str, Any]:
        """Generate migration statistics."""
        test_files = list(test_directory.rglob("test_*.py")) + list(
            test_directory.rglob("*_test.py")
        )

        # Analyze test files for migration patterns
        migration_stats = {
            "total_migrated_files": len(test_files),
            "migration_patterns": self._analyze_migration_patterns(test_files),
            "complexity_analysis": self._analyze_test_complexity(test_files),
        }

        return migration_stats

    def _generate_coverage_report(self, test_directory: Path) -> dict[str, Any]:
        """Generate test coverage report."""
        return {
            "coverage_available": False,
            "message": "Run tests with coverage to populate coverage data",
        }

    def _generate_diagnostic_report(self, test_directory: Path) -> dict[str, Any]:
        """Generate diagnostic report."""
        return {
            "diagnostics_available": False,
            "message": "Run migration with diagnostics enabled to populate diagnostic data",
        }

    def _analyze_migration_patterns(self, test_files: list[Path]) -> dict[str, Any]:
        """Analyze migration patterns in test files."""
        patterns = {
            "pytest_fixtures": 0,
            "assert_statements": 0,
            "mock_usage": 0,
            "integration_tests": 0,
        }

        for test_file in test_files:
            try:
                content = test_file.read_text()

                # Count pytest patterns
                if "@pytest.fixture" in content:
                    patterns["pytest_fixtures"] += content.count("@pytest.fixture")

                if "assert " in content:
                    patterns["assert_statements"] += content.count("assert ")

                if "mock" in content.lower():
                    patterns["mock_usage"] += 1

                if "@pytest.mark.integration" in content:
                    patterns["integration_tests"] += 1

            except Exception:
                continue

        return patterns

    def _analyze_test_complexity(self, test_files: list[Path]) -> dict[str, Any]:
        """Analyze test complexity metrics."""
        complexity = {
            "total_lines": 0,
            "average_file_size": 0,
            "largest_file": None,
            "smallest_file": None,
        }

        file_sizes = []

        for test_file in test_files:
            try:
                content = test_file.read_text()
                lines = len(content.splitlines())
                file_sizes.append(lines)
                complexity["total_lines"] += lines

                if complexity["largest_file"] is None or lines > file_sizes[0]:
                    complexity["largest_file"] = {"file": str(test_file), "lines": lines}

                if complexity["smallest_file"] is None or lines < file_sizes[0]:
                    complexity["smallest_file"] = {"file": str(test_file), "lines": lines}

            except Exception:
                continue

        if file_sizes:
            complexity["average_file_size"] = sum(file_sizes) / len(file_sizes)

        return complexity

    def _save_html_report(
        self, report_data: dict[str, Any], output_file: Path, template_file: Path | None
    ):
        """Save HTML report."""
        if template_file and template_file.exists():
            template_content = template_file.read_text()
        else:
            template_content = self._get_default_html_template()

        # Simple template substitution
        html_content = template_content.replace(
            "{{REPORT_DATA}}", json.dumps(report_data, indent=2, default=str)
        )

        html_content = html_content.replace(
            "{{GENERATED_AT}}", report_data["metadata"]["generated_at"]
        )

        output_file.write_text(html_content)

    def _save_json_report(self, report_data: dict[str, Any], output_file: Path):
        """Save JSON report."""
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def _save_yaml_report(self, report_data: dict[str, Any], output_file: Path):
        """Save YAML report."""
        try:
            import yaml

            with open(output_file, "w") as f:
                yaml.dump(report_data, f, default_flow_style=False)
        except ImportError as e:
            raise ImportError("PyYAML is required for YAML output") from e

    def _save_text_report(self, report_data: dict[str, Any], output_file: Path):
        """Save text report."""
        content = self._format_text_report(report_data)
        output_file.write_text(content)

    def _save_pdf_report(self, report_data: dict[str, Any], output_file: Path):
        """Save PDF report."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

            doc = SimpleDocTemplate(str(output_file), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title = Paragraph("Test Migration Report", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 12))

            # Summary
            summary_text = self._format_text_report(report_data)
            summary = Paragraph(summary_text.replace("\n", "<br/>"), styles["Normal"])
            story.append(summary)

            doc.build(story)

        except ImportError as e:
            raise ImportError("ReportLab is required for PDF output") from e

    def _format_text_report(self, report_data: dict[str, Any]) -> str:
        """Format report data as text."""
        lines = []
        lines.append("Test Migration Report")
        lines.append("=" * 50)
        lines.append(f"Generated: {report_data['metadata']['generated_at']}")
        lines.append("")

        # Summary
        summary = report_data.get("summary", {})
        lines.append("Summary:")
        lines.append(f"  Test Directory: {summary.get('test_directory', 'N/A')}")
        lines.append(f"  Total Test Files: {summary.get('total_test_files', 0)}")
        lines.append("")

        # Migration Statistics
        stats = report_data.get("migration_statistics", {})
        lines.append("Migration Statistics:")
        lines.append(f"  Migrated Files: {stats.get('total_migrated_files', 0)}")

        patterns = stats.get("migration_patterns", {})
        if patterns:
            lines.append("  Migration Patterns:")
            for pattern, count in patterns.items():
                lines.append(f"    {pattern}: {count}")

        complexity = stats.get("complexity_analysis", {})
        if complexity:
            lines.append("  Complexity Analysis:")
            lines.append(f"    Total Lines: {complexity.get('total_lines', 0)}")
            lines.append(
                f"    Average File Size: {complexity.get('average_file_size', 0):.1f} lines"
            )

        return "\n".join(lines)

    def _get_default_html_template(self) -> str:
        """Get default HTML template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Test Migration Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .data { background-color: #f9f9f9; padding: 10px; border-radius: 3px; }
        pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Migration Report</h1>
        <p>Generated: {{GENERATED_AT}}</p>
    </div>

    <div class="section">
        <h2>Report Data</h2>
        <div class="data">
            <pre>{{REPORT_DATA}}</pre>
        </div>
    </div>
</body>
</html>
"""

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.datetime.now().isoformat()


class ReportFormatter:
    """Utility class for formatting report data."""

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @staticmethod
    def format_percentage(value: float, total: float) -> str:
        """Format percentage with proper handling of zero division."""
        if total == 0:
            return "0.0%"
        return f"{(value / total) * 100:.1f}%"

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
