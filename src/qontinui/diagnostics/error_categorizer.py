"""Error categorization and fix suggestions for diagnostics.

Analyzes diagnostic results and provides categorized recommendations.
"""

from typing import Any


class ErrorCategory:
    """Categories for diagnostic errors."""

    MISSING_LIBRARY = "missing_library"
    MISSING_PATH = "missing_path"
    LOAD_FAILURE = "load_failure"
    PERFORMANCE_ISSUE = "performance_issue"
    NO_IMAGES = "no_images"
    DISPLAY_ISSUE = "display_issue"


class ErrorCategorizer:
    """Categorizes errors and issues from diagnostic results."""

    @staticmethod
    def categorize_library_issues(libraries: dict[str, Any]) -> list[dict[str, str]]:
        """Categorize missing or problematic libraries.

        Args:
            libraries: Library availability information

        Returns:
            List of library issues with category and description
        """
        issues = []

        if not libraries.get("PIL/Pillow"):
            issues.append(
                {
                    "category": ErrorCategory.MISSING_LIBRARY,
                    "library": "Pillow",
                    "severity": "critical",
                    "description": "PIL/Pillow is required for image processing",
                }
            )

        if not libraries.get("OpenCV"):
            issues.append(
                {
                    "category": ErrorCategory.MISSING_LIBRARY,
                    "library": "OpenCV",
                    "severity": "optional",
                    "description": "OpenCV provides advanced image operations",
                }
            )

        return issues

    @staticmethod
    def categorize_path_issues(paths: dict[str, Any]) -> list[dict[str, str]]:
        """Categorize path configuration issues.

        Args:
            paths: Path configuration information

        Returns:
            List of path issues with category and description
        """
        issues = []

        if not paths.get("image_path_valid"):
            issues.append(
                {
                    "category": ErrorCategory.MISSING_PATH,
                    "path": paths.get("image_path", ""),
                    "severity": "critical",
                    "description": f"Image path does not exist or is not accessible: {paths.get('image_path_reason', '')}",
                }
            )

        if not paths.get("screenshot_path_valid"):
            issues.append(
                {
                    "category": ErrorCategory.MISSING_PATH,
                    "path": paths.get("screenshot_path", ""),
                    "severity": "warning",
                    "description": f"Screenshot path does not exist or is not accessible: {paths.get('screenshot_path_reason', '')}",
                }
            )

        return issues

    @staticmethod
    def categorize_performance_issues(performance: dict[str, Any]) -> list[dict[str, Any]]:
        """Categorize performance-related issues.

        Args:
            performance: Performance statistics

        Returns:
            List of performance issues
        """
        issues = []

        avg_load_time = performance.get("average_load_time_ms", 0)
        if avg_load_time > 100:
            issues.append(
                {
                    "category": ErrorCategory.PERFORMANCE_ISSUE,
                    "metric": "load_time",
                    "value": avg_load_time,
                    "threshold": 100,
                    "severity": "warning",
                    "description": f"High average load time: {avg_load_time:.1f}ms (threshold: 100ms)",
                }
            )

        failed_loads = performance.get("failed_loads", 0)
        if failed_loads > 0:
            issues.append(
                {
                    "category": ErrorCategory.LOAD_FAILURE,
                    "metric": "failed_loads",
                    "value": failed_loads,
                    "severity": "error",
                    "description": f"{failed_loads} image load failures detected",
                }
            )

        return issues

    @staticmethod
    def categorize_directory_issues(directories: dict[str, Any]) -> list[dict[str, Any]]:
        """Categorize directory structure issues.

        Args:
            directories: Directory analysis results

        Returns:
            List of directory issues
        """
        issues = []

        img_dir = directories.get("image_directory", {})
        if img_dir.get("exists") and img_dir.get("image_files", 0) == 0:
            issues.append(
                {
                    "category": ErrorCategory.NO_IMAGES,
                    "path": img_dir.get("path", ""),
                    "severity": "warning",
                    "description": "No images found in image directory",
                }
            )

        return issues


class FixSuggester:
    """Suggests fixes for categorized errors."""

    FIX_TEMPLATES = {
        "Pillow": "Install Pillow for image processing: pip install Pillow",
        "OpenCV": "Consider installing OpenCV for advanced image operations: pip install opencv-python",
    }

    @staticmethod
    def _suggest_fix(issue: dict[str, Any]) -> str:
        """Generate fix suggestion for any issue type.

        Args:
            issue: Issue information

        Returns:
            Fix suggestion
        """
        category = issue.get("category", "")

        if category == ErrorCategory.MISSING_LIBRARY:
            library = issue.get("library", "")
            return FixSuggester.FIX_TEMPLATES.get(
                library, f"Install {library}: pip install {library.lower()}"
            )

        if category == ErrorCategory.MISSING_PATH:
            return f"Create directory: {issue.get('path', '')}"

        if category == ErrorCategory.PERFORMANCE_ISSUE:
            value = issue.get("value", 0)
            return (
                f"High average load time ({value:.1f}ms) - consider image optimization or caching"
            )

        if category == ErrorCategory.LOAD_FAILURE:
            value = issue.get("value", 0)
            return f"{value} image load failures - check file paths and formats"

        if category == ErrorCategory.NO_IMAGES:
            return "Add image resources to the image directory"

        return "Check diagnostic results and address issues"

    @staticmethod
    def generate_all_suggestions(categorized_issues: dict[str, list[dict[str, Any]]]) -> list[str]:
        """Generate all fix suggestions from categorized issues.

        Args:
            categorized_issues: Dictionary of categorized issues

        Returns:
            List of fix suggestions
        """
        suggestions = []

        for issue_list in categorized_issues.values():
            for issue in issue_list:
                suggestions.append(FixSuggester._suggest_fix(issue))

        return suggestions if suggestions else ["All diagnostics passed successfully"]
