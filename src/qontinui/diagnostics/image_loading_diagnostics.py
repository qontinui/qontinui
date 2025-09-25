"""Image loading diagnostics - ported from Qontinui framework.

Diagnostic tools for testing and troubleshooting image loading capabilities.
"""

import logging
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of an image load attempt."""

    success: bool
    """Whether the load succeeded."""

    image_name: str
    """Name of the image attempted."""

    loaded_from: str | None = None
    """Source location (cache, file, classpath, etc.)."""

    load_time_ms: float | None = None
    """Time taken to load in milliseconds."""

    failure_reason: str | None = None
    """Reason for failure if not successful."""

    image_dimensions: tuple[int, int] | None = None
    """Width and height if loaded successfully."""


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
    """Diagnostic runner for testing image loading capabilities.

    Port of ImageLoadingDiagnosticsRunner from Qontinui framework.

    Provides comprehensive diagnostics for:
    - Environment information
    - Path configuration
    - Directory structure
    - Image loading tests
    - Performance analysis
    - Recommendations
    """

    def __init__(self, config: DiagnosticsConfig | None = None):
        """Initialize diagnostics.

        Args:
            config: Diagnostics configuration
        """
        self.config = config or DiagnosticsConfig()
        self.load_history: dict[str, LoadResult] = {}
        self.performance_stats: dict[str, float] = defaultdict(float)

    def run_diagnostics(self) -> dict[str, Any]:
        """Run complete diagnostics suite.

        Returns:
            Dictionary with all diagnostic results
        """
        logger.info("=== Qontinui Image Loading Diagnostics ===")

        results = {}

        # 1. Environment Information
        results["environment"] = self._get_environment_info()
        self._print_environment_info(results["environment"])

        # 2. Path Configuration
        results["paths"] = self._get_path_configuration()
        self._print_path_configuration(results["paths"])

        # 3. Directory Structure
        if self.config.check_paths:
            results["directories"] = self._analyze_directory_structure()
            self._print_directory_structure(results["directories"])

        # 4. Test Specific Images
        if self.config.test_images:
            results["specific_tests"] = self._test_specific_images()
            self._print_test_results(results["specific_tests"])

        # 5. Test All Found Images
        if self.config.test_all_images:
            results["all_tests"] = self._test_all_found_images()
            self._print_all_test_results(results["all_tests"])

        # 6. Performance Report
        if self.config.performance_test:
            results["performance"] = self._generate_performance_report()
            self._print_performance_report(results["performance"])

        # 7. Recommendations
        results["recommendations"] = self._generate_recommendations(results)
        self._print_recommendations(results["recommendations"])

        logger.info("=== Image Loading Diagnostics Complete ===")

        return results

    def _get_environment_info(self) -> dict[str, Any]:
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
            elif platform.system() == "Windows":
                return True
            elif platform.system() == "Darwin":
                return True
            return False
        except Exception:
            return False

    def _check_image_libraries(self) -> dict[str, bool]:
        """Check available image processing libraries.

        Returns:
            Dictionary of library availability
        """
        libraries = {}

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
            libraries["skimage_version"] = skimage.__version__
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

    def _get_path_configuration(self) -> dict[str, Any]:
        """Get path configuration.

        Returns:
            Path configuration details
        """
        from ..config import get_settings

        settings = get_settings()

        paths = {
            "image_path": settings.image_path,
            "screenshot_path": settings.screenshot_path,
            "python_path": sys.path[:5],  # First 5 paths
            "current_directory": os.getcwd(),
            "home_directory": str(Path.home()),
        }

        # Check if paths exist
        paths["image_path_exists"] = os.path.exists(settings.image_path)
        paths["screenshot_path_exists"] = os.path.exists(settings.screenshot_path)

        return paths

    def _analyze_directory_structure(self) -> dict[str, Any]:
        """Analyze directory structure for images.

        Returns:
            Directory analysis results
        """
        from ..config import get_settings

        settings = get_settings()
        results = {}

        # Analyze image directory
        image_dir = Path(settings.image_path)
        if image_dir.exists():
            results["image_directory"] = self._analyze_directory(image_dir)
        else:
            results["image_directory"] = {"exists": False, "path": str(image_dir)}

        # Analyze screenshot directory
        screenshot_dir = Path(settings.screenshot_path)
        if screenshot_dir.exists():
            results["screenshot_directory"] = self._analyze_directory(screenshot_dir)
        else:
            results["screenshot_directory"] = {"exists": False, "path": str(screenshot_dir)}

        return results

    def _analyze_directory(self, directory: Path) -> dict[str, Any]:
        """Analyze a single directory.

        Args:
            directory: Directory to analyze

        Returns:
            Directory analysis
        """
        analysis = {
            "exists": True,
            "path": str(directory),
            "total_files": 0,
            "image_files": 0,
            "subdirectories": 0,
            "file_types": defaultdict(int),
            "total_size_mb": 0.0,
        }

        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

        for item in directory.rglob("*"):
            if item.is_file():
                analysis["total_files"] += 1
                extension = item.suffix.lower()
                analysis["file_types"][extension] += 1

                if extension in image_extensions:
                    analysis["image_files"] += 1

                try:
                    analysis["total_size_mb"] += item.stat().st_size / (1024 * 1024)
                except Exception:
                    pass

            elif item.is_dir():
                analysis["subdirectories"] += 1

        analysis["file_types"] = dict(analysis["file_types"])

        return analysis

    def _test_specific_images(self) -> list[LoadResult]:
        """Test loading specific images.

        Returns:
            List of load results
        """
        results = []

        for image_name in self.config.test_images:
            result = self._test_image_loading(image_name)
            results.append(result)
            self.load_history[image_name] = result

        return results

    def _test_image_loading(self, image_name: str) -> LoadResult:
        """Test loading a single image.

        Args:
            image_name: Name of image to test

        Returns:
            Load result
        """
        start_time = time.time()

        try:
            # Try to load with PIL
            from PIL import Image

            # Try different paths
            paths_to_try = [
                image_name,
                f"images/{image_name}",
                f"resources/images/{image_name}",
                Path(image_name),
            ]

            for path in paths_to_try:
                try:
                    img = Image.open(path)
                    load_time_ms = (time.time() - start_time) * 1000

                    return LoadResult(
                        success=True,
                        image_name=image_name,
                        loaded_from=str(path),
                        load_time_ms=load_time_ms,
                        image_dimensions=img.size,
                    )
                except Exception:
                    continue

            # If all paths failed
            return LoadResult(
                success=False,
                image_name=image_name,
                failure_reason="Image not found in any configured path",
            )

        except Exception as e:
            return LoadResult(success=False, image_name=image_name, failure_reason=str(e))

    def _test_all_found_images(self) -> dict[str, Any]:
        """Test all images found in configured directories.

        Returns:
            Test results summary
        """
        from ..config import get_settings

        settings = get_settings()
        image_dir = Path(settings.image_path)

        if not image_dir.exists():
            return {"error": "Image directory does not exist"}

        all_images = []
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

        for item in image_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in image_extensions:
                all_images.append(str(item))

        successful = 0
        failed = 0
        total_time = 0.0

        for image_path in all_images:
            result = self._test_image_loading(image_path)
            if result.success:
                successful += 1
                if result.load_time_ms:
                    total_time += result.load_time_ms
            else:
                failed += 1
                if self.config.verbose:
                    logger.warning(f"Failed to load: {image_path} - {result.failure_reason}")

        return {
            "total_images": len(all_images),
            "successful": successful,
            "failed": failed,
            "average_load_time_ms": total_time / successful if successful > 0 else 0,
            "success_rate": (successful / len(all_images) * 100) if all_images else 0,
        }

    def _generate_performance_report(self) -> dict[str, Any]:
        """Generate performance report.

        Returns:
            Performance statistics
        """
        if not self.load_history:
            return {"message": "No images loaded yet"}

        total_attempts = len(self.load_history)
        successful = sum(1 for r in self.load_history.values() if r.success)
        failed = total_attempts - successful

        load_times = [
            r.load_time_ms for r in self.load_history.values() if r.load_time_ms is not None
        ]

        report = {
            "total_attempts": total_attempts,
            "successful_loads": successful,
            "failed_loads": failed,
            "success_rate": (successful / total_attempts * 100) if total_attempts > 0 else 0,
        }

        if load_times:
            report["average_load_time_ms"] = sum(load_times) / len(load_times)
            report["min_load_time_ms"] = min(load_times)
            report["max_load_time_ms"] = max(load_times)

        return report

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on diagnostics.

        Args:
            results: All diagnostic results

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check environment
        env = results.get("environment", {})
        libs = env.get("image_libraries", {})

        if not libs.get("PIL/Pillow"):
            recommendations.append("Install Pillow for image processing: pip install Pillow")

        if not libs.get("OpenCV"):
            recommendations.append(
                "Consider installing OpenCV for advanced image operations: pip install opencv-python"
            )

        # Check paths
        paths = results.get("paths", {})
        if not paths.get("image_path_exists"):
            recommendations.append(f"Create image directory: {paths.get('image_path')}")

        # Check performance
        perf = results.get("performance", {})
        if perf.get("failed_loads", 0) > 0:
            recommendations.append(
                f"{perf['failed_loads']} image load failures detected - check file paths and formats"
            )

        avg_load_time = perf.get("average_load_time_ms", 0)
        if avg_load_time > 100:
            recommendations.append(
                f"High average load time ({avg_load_time:.1f}ms) - consider image optimization"
            )

        # Check directories
        dirs = results.get("directories", {})
        img_dir = dirs.get("image_directory", {})
        if img_dir.get("image_files", 0) == 0 and img_dir.get("exists"):
            recommendations.append("No images found in image directory - add image resources")

        if not recommendations:
            recommendations.append("✓ All diagnostics passed successfully!")

        return recommendations

    def _print_environment_info(self, info: dict[str, Any]) -> None:
        """Print environment information."""
        if not self.config.verbose:
            return

        logger.info("\n=== Environment Information ===")
        logger.info(f"Working Directory: {info['working_directory']}")
        logger.info(f"Python Version: {info['python_version'].split()[0]}")
        logger.info(f"Platform: {info['platform']}")
        logger.info(f"Display Available: {info['display_available']}")

        logger.info("Image Libraries:")
        for lib, available in info["image_libraries"].items():
            if not lib.endswith("_version"):
                status = "✓" if available else "✗"
                version = info["image_libraries"].get(f"{lib}_version", "")
                logger.info(f"  {status} {lib} {version}")

    def _print_path_configuration(self, paths: dict[str, Any]) -> None:
        """Print path configuration."""
        if not self.config.verbose:
            return

        logger.info("\n=== Path Configuration ===")
        logger.info(f"Image Path: {paths['image_path']} (exists: {paths['image_path_exists']})")
        logger.info(
            f"Screenshot Path: {paths['screenshot_path']} (exists: {paths['screenshot_path_exists']})"
        )

    def _print_directory_structure(self, dirs: dict[str, Any]) -> None:
        """Print directory structure analysis."""
        if not self.config.verbose:
            return

        logger.info("\n=== Directory Structure ===")

        for name, analysis in dirs.items():
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

    def _print_test_results(self, results: list[LoadResult]) -> None:
        """Print specific image test results."""
        if not self.config.verbose or not results:
            return

        logger.info("\n=== Specific Image Tests ===")

        for result in results:
            if result.success:
                logger.info(f"✓ {result.image_name}")
                logger.info(f"  Loaded from: {result.loaded_from}")
                logger.info(f"  Load time: {result.load_time_ms:.2f}ms")
                if result.image_dimensions:
                    logger.info(
                        f"  Dimensions: {result.image_dimensions[0]}x{result.image_dimensions[1]}"
                    )
            else:
                logger.warning(f"✗ {result.image_name}")
                logger.warning(f"  Reason: {result.failure_reason}")

    def _print_all_test_results(self, results: dict[str, Any]) -> None:
        """Print all image test results."""
        if not self.config.verbose:
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

    def _print_performance_report(self, report: dict[str, Any]) -> None:
        """Print performance report."""
        if not self.config.verbose:
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

    def _print_recommendations(self, recommendations: list[str]) -> None:
        """Print recommendations."""
        logger.info("\n=== Recommendations ===")

        for rec in recommendations:
            if rec.startswith("✓"):
                logger.info(rec)
            else:
                logger.info(f"• {rec}")


def run_diagnostics(config: DiagnosticsConfig | None = None) -> dict[str, Any]:
    """Run image loading diagnostics.

    Args:
        config: Optional configuration

    Returns:
        Diagnostic results
    """
    diagnostics = ImageLoadingDiagnostics(config)
    return diagnostics.run_diagnostics()
