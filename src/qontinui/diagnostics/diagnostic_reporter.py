"""Diagnostic report generation and formatting.

Handles formatting and printing of diagnostic results.
"""

import logging
from typing import Any

from .image_loader import LoadResult

logger = logging.getLogger(__name__)


class DiagnosticReporter:
    """Generates and formats diagnostic reports."""

    @staticmethod
    def print_test_results(results: list[LoadResult], verbose: bool = True) -> None:
        """Print specific image test results.

        Args:
            results: Load results to print
            verbose: Whether to print verbose output
        """
        if not verbose or not results:
            return

        logger.info("\n=== Specific Image Tests ===")

        for result in results:
            if result.success:
                logger.info(f"SUCCESS: {result.image_name}")
                logger.info(f"  Loaded from: {result.loaded_from}")
                logger.info(f"  Load time: {result.load_time_ms:.2f}ms")
                if result.image_dimensions:
                    logger.info(
                        f"  Dimensions: {result.image_dimensions[0]}x{result.image_dimensions[1]}"
                    )
            else:
                logger.warning(f"FAILED: {result.image_name}")
                logger.warning(f"  Reason: {result.failure_reason}")

    @staticmethod
    def print_all_test_results(results: dict[str, Any], verbose: bool = True) -> None:
        """Print all image test results.

        Args:
            results: All test results
            verbose: Whether to print verbose output
        """
        if not verbose:
            return

        logger.info("\n=== All Image Tests ===")

        if "error" in results:
            logger.error(results["error"])
        else:
            logger.info(f"Total Images: {results['total_images']}")
            logger.info(f"Successful: {results['successful']}")
            logger.info(f"Failed: {results['failed']}")
            logger.info(f"Success Rate: {results['success_rate']:.1f}%")
            logger.info(f"Average Load Time: {results['average_load_time_ms']:.2f}ms")

    @staticmethod
    def print_performance_report(report: dict[str, Any], verbose: bool = True) -> None:
        """Print performance report.

        Args:
            report: Performance statistics
            verbose: Whether to print verbose output
        """
        if not verbose:
            return

        logger.info("\n=== Performance Report ===")

        if "message" in report:
            logger.info(report["message"])
        else:
            logger.info(f"Total Attempts: {report['total_attempts']}")
            logger.info(f"Successful: {report['successful_loads']}")
            logger.info(f"Failed: {report['failed_loads']}")
            logger.info(f"Success Rate: {report['success_rate']:.1f}%")

            if "average_load_time_ms" in report:
                logger.info(f"Average Load Time: {report['average_load_time_ms']:.2f}ms")
                logger.info(f"Min Load Time: {report['min_load_time_ms']:.2f}ms")
                logger.info(f"Max Load Time: {report['max_load_time_ms']:.2f}ms")

    @staticmethod
    def print_recommendations(recommendations: list[str]) -> None:
        """Print recommendations.

        Args:
            recommendations: List of recommendations
        """
        logger.info("\n=== Recommendations ===")

        for rec in recommendations:
            if "successfully" in rec.lower():
                logger.info(rec)
            else:
                logger.info(f"- {rec}")

    @staticmethod
    def generate_summary(results: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of diagnostic results.

        Args:
            results: Complete diagnostic results

        Returns:
            Summary dictionary
        """
        summary = {
            "environment_ok": True,
            "paths_ok": True,
            "performance_ok": True,
            "issues": [],
        }

        # Check environment
        env = results.get("environment", {})
        libs = env.get("image_libraries", {})
        if not libs.get("PIL/Pillow"):
            summary["environment_ok"] = False
            summary["issues"].append("PIL/Pillow not available")  # type: ignore[attr-defined]

        # Check paths
        paths = results.get("paths", {})
        if not paths.get("image_path_valid"):
            summary["paths_ok"] = False
            summary["issues"].append("Image path not valid")  # type: ignore[attr-defined]

        # Check performance
        perf = results.get("performance", {})
        if perf.get("failed_loads", 0) > 0:
            summary["performance_ok"] = False
            summary["issues"].append(f"{perf['failed_loads']} load failures")  # type: ignore[attr-defined]

        summary["overall_ok"] = (
            summary["environment_ok"] and summary["paths_ok"] and summary["performance_ok"]
        )

        return summary

    @staticmethod
    def format_results_json(results: dict[str, Any]) -> str:
        """Format results as JSON string.

        Args:
            results: Results to format

        Returns:
            JSON formatted string
        """
        import json

        return json.dumps(results, indent=2, default=str)

    @staticmethod
    def print_header(title: str) -> None:
        """Print a formatted header.

        Args:
            title: Header title
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{title:^60}")
        logger.info(f"{'=' * 60}")

    @staticmethod
    def print_footer() -> None:
        """Print a formatted footer."""
        logger.info(f"{'=' * 60}\n")
