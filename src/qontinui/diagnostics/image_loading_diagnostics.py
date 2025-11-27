"""Image loading diagnostics - refactored facade.

Diagnostic tools for testing and troubleshooting image loading capabilities.
Uses Strategy Pattern for modular diagnostic operations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_settings
from .diagnostic_reporter import DiagnosticReporter
from .diagnostic_strategies import (
    DirectoryStructureStrategy,
    EnvironmentStrategy,
    PathConfigurationStrategy,
)
from .error_categorizer import ErrorCategorizer, FixSuggester
from .image_loader import ImageLoader, LoadResult

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics run."""

    test_images: list[str] = field(default_factory=list)
    """Specific images to test."""

    verbose: bool = True
    """Enable verbose output."""

    test_all_images: bool = False
    """Test all found images."""

    performance_test: bool = True
    """Run performance tests."""

    check_paths: bool = True
    """Check configured paths."""


class ImageLoadingDiagnostics:
    """Diagnostic facade for testing image loading capabilities.

    Coordinates multiple diagnostic strategies to provide comprehensive
    diagnostics for:
    - Environment information
    - Path configuration
    - Directory structure
    - Image loading tests
    - Performance analysis
    - Recommendations
    """

    def __init__(self, config: DiagnosticsConfig | None = None) -> None:
        """Initialize diagnostics.

        Args:
            config: Diagnostics configuration
        """
        self.config = config or DiagnosticsConfig()
        self.image_loader = ImageLoader()
        self.reporter = DiagnosticReporter()

        # Initialize strategies
        self.env_strategy = EnvironmentStrategy()
        self.path_strategy = PathConfigurationStrategy()
        self.dir_strategy = DirectoryStructureStrategy()

    def run_diagnostics(self) -> dict[str, Any]:
        """Run complete diagnostics suite.

        Returns:
            Dictionary with all diagnostic results
        """
        self.reporter.print_header("Qontinui Image Loading Diagnostics")

        results: dict[str, Any] = {}

        # Run diagnostic strategies
        results["environment"] = self.env_strategy.run()
        self.env_strategy.print_results(results["environment"], self.config.verbose)

        results["paths"] = self.path_strategy.run()
        self.path_strategy.print_results(results["paths"], self.config.verbose)

        if self.config.check_paths:
            results["directories"] = self.dir_strategy.run()
            self.dir_strategy.print_results(results["directories"], self.config.verbose)

        # Test specific images
        if self.config.test_images:
            specific_tests = self.image_loader.load_multiple_images(self.config.test_images)
            results["specific_tests"] = specific_tests
            self.reporter.print_test_results(specific_tests, self.config.verbose)

        # Test all found images
        if self.config.test_all_images:
            settings = get_settings()
            image_dir = Path(settings.core.image_path)
            all_tests = self.image_loader.load_all_images_in_directory(image_dir)
            results["all_tests"] = all_tests
            self.reporter.print_all_test_results(all_tests, self.config.verbose)

        # Performance report
        if self.config.performance_test:
            performance = self.image_loader.get_load_statistics()
            results["performance"] = performance
            self.reporter.print_performance_report(performance, self.config.verbose)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations
        self.reporter.print_recommendations(recommendations)

        self.reporter.print_footer()

        return results

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on diagnostics.

        Args:
            results: All diagnostic results

        Returns:
            List of recommendations
        """
        categorized_issues: dict[str, list[dict[str, Any]]] = {}

        # Categorize environment issues
        env = results.get("environment", {})
        libs = env.get("image_libraries", {})
        categorized_issues["library_issues"] = ErrorCategorizer.categorize_library_issues(libs)

        # Categorize path issues
        paths = results.get("paths", {})
        categorized_issues["path_issues"] = ErrorCategorizer.categorize_path_issues(paths)

        # Categorize performance issues
        perf = results.get("performance", {})
        if not isinstance(perf, dict) or "message" not in perf:
            categorized_issues["performance_issues"] = (
                ErrorCategorizer.categorize_performance_issues(perf)
            )
        else:
            categorized_issues["performance_issues"] = []

        # Categorize directory issues
        dirs = results.get("directories", {})
        if dirs:
            categorized_issues["directory_issues"] = ErrorCategorizer.categorize_directory_issues(
                dirs
            )
        else:
            categorized_issues["directory_issues"] = []

        # Generate fix suggestions
        return FixSuggester.generate_all_suggestions(categorized_issues)


def run_diagnostics(config: DiagnosticsConfig | None = None) -> dict[str, Any]:
    """Run image loading diagnostics.

    Args:
        config: Optional configuration

    Returns:
        Diagnostic results
    """
    diagnostics = ImageLoadingDiagnostics(config)
    return diagnostics.run_diagnostics()


# Re-export for backward compatibility
__all__ = ["ImageLoadingDiagnostics", "DiagnosticsConfig", "LoadResult", "run_diagnostics"]
