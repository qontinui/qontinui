"""Strategy Pattern implementation for diagnostic operations.

Defines base strategy classes and concrete implementations for different
types of diagnostic checks.
"""

import logging
import os
import platform
import sys
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class DiagnosticStrategy(ABC):
    """Base strategy for diagnostic operations."""

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Run the diagnostic check.

        Returns:
            Diagnostic results
        """
        pass

    @abstractmethod
    def print_results(self, results: dict[str, Any], verbose: bool = True) -> None:
        """Print diagnostic results.

        Args:
            results: Results to print
            verbose: Whether to print verbose output
        """
        pass


class EnvironmentStrategy(DiagnosticStrategy):
    """Strategy for environment diagnostics."""

    def run(self) -> dict[str, Any]:
        """Get environment information.

        Returns:
            Environment details
        """
        info = {
            "working_directory": os.getcwd(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "display_available": self._check_display(),
            "image_libraries": self._check_image_libraries(),
        }

        return info

    def _check_display(self) -> bool:
        """Check if display is available.

        Returns:
            True if display available
        """
        try:
            if platform.system() == "Linux":
                return "DISPLAY" in os.environ
            elif platform.system() in ("Windows", "Darwin"):
                return True
            return False
        except (OSError, RuntimeError):
            return False

    def _check_image_libraries(self) -> dict[str, Any]:
        """Check available image processing libraries.

        Returns:
            Dictionary of library availability
        """
        libraries: dict[str, Any] = {}

        # Check PIL/Pillow
        try:
            import PIL

            libraries["PIL/Pillow"] = True
            libraries["PIL_version"] = PIL.__version__
        except ImportError:
            libraries["PIL/Pillow"] = False

        # Check OpenCV
        try:
            import cv2

            libraries["OpenCV"] = True
            libraries["OpenCV_version"] = cv2.__version__
        except ImportError:
            libraries["OpenCV"] = False

        # Check scikit-image
        try:
            import skimage

            libraries["scikit-image"] = True
            libraries["skimage_version"] = skimage.__version__  # type: ignore[attr-defined]
        except ImportError:
            libraries["scikit-image"] = False

        # Check matplotlib
        try:
            import matplotlib

            libraries["matplotlib"] = True
            libraries["matplotlib_version"] = matplotlib.__version__
        except ImportError:
            libraries["matplotlib"] = False

        return libraries

    def print_results(self, results: dict[str, Any], verbose: bool = True) -> None:
        """Print environment information.

        Args:
            results: Environment information
            verbose: Whether to print verbose output
        """
        if not verbose:
            return

        logger.info("\n=== Environment Information ===")
        logger.info(f"Working Directory: {results['working_directory']}")
        logger.info(f"Python Version: {results['python_version'].split()[0]}")
        logger.info(f"Platform: {results['platform']}")
        logger.info(f"Display Available: {results['display_available']}")

        logger.info("Image Libraries:")
        for lib, available in results["image_libraries"].items():
            if not lib.endswith("_version"):
                status = "Available" if available else "Missing"
                version = results["image_libraries"].get(f"{lib}_version", "")
                logger.info(f"  {lib}: {status} {version}")


class PathConfigurationStrategy(DiagnosticStrategy):
    """Strategy for path configuration diagnostics."""

    def run(self) -> dict[str, Any]:
        """Get path configuration.

        Returns:
            Path configuration details
        """
        from .image_validator import ConfigurationValidator

        return ConfigurationValidator.validate_path_configuration()

    def print_results(self, results: dict[str, Any], verbose: bool = True) -> None:
        """Print path configuration.

        Args:
            results: Path configuration
            verbose: Whether to print verbose output
        """
        if not verbose:
            return

        logger.info("\n=== Path Configuration ===")
        logger.info(f"Image Path: {results['image_path']} (valid: {results['image_path_valid']})")
        if not results["image_path_valid"]:
            logger.info(f"  Reason: {results['image_path_reason']}")

        logger.info(
            f"Screenshot Path: {results['screenshot_path']} (valid: {results['screenshot_path_valid']})"
        )
        if not results["screenshot_path_valid"]:
            logger.info(f"  Reason: {results['screenshot_path_reason']}")


class DirectoryStructureStrategy(DiagnosticStrategy):
    """Strategy for directory structure diagnostics."""

    def run(self) -> dict[str, Any]:
        """Analyze directory structure.

        Returns:
            Directory analysis results
        """
        from .image_validator import DirectoryAnalyzer

        return DirectoryAnalyzer.analyze_configured_directories()

    def print_results(self, results: dict[str, Any], verbose: bool = True) -> None:
        """Print directory structure analysis.

        Args:
            results: Directory analysis
            verbose: Whether to print verbose output
        """
        if not verbose:
            return

        logger.info("\n=== Directory Structure ===")

        for name, analysis in results.items():
            logger.info(f"\n{name}:")
            if analysis["exists"]:
                logger.info(f"  Path: {analysis['path']}")
                logger.info(f"  Total Files: {analysis['total_files']}")
                logger.info(f"  Image Files: {analysis['image_files']}")
                logger.info(f"  Subdirectories: {analysis['subdirectories']}")
                logger.info(f"  Total Size: {analysis['total_size_mb']:.2f} MB")
                if analysis.get("file_types"):
                    logger.info("  File Types:")
                    for ext, count in analysis["file_types"].items():
                        logger.info(f"    {ext}: {count}")
            else:
                logger.info(f"  Path: {analysis['path']} (NOT FOUND)")
