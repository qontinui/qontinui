"""Image path and file validation for diagnostics.

Handles validation of image paths, directories, and file accessibility.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import get_settings


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    """Whether the validation passed."""

    path: str
    """Path that was validated."""

    reason: str | None = None
    """Reason for failure if not valid."""

    details: dict[str, Any] | None = None
    """Additional validation details."""


class PathValidator:
    """Validates paths for image loading."""

    @staticmethod
    def validate_image_path(image_name: str) -> list[str | Path]:
        """Generate candidate paths for an image.

        Args:
            image_name: Name of the image file

        Returns:
            List of paths to try for loading
        """
        return [
            image_name,
            f"images/{image_name}",
            f"resources/images/{image_name}",
            Path(image_name),
        ]

    @staticmethod
    def validate_directory(directory: Path) -> ValidationResult:
        """Validate a directory exists and is accessible.

        Args:
            directory: Directory to validate

        Returns:
            Validation result
        """
        if not directory.exists():
            return ValidationResult(
                valid=False, path=str(directory), reason="Directory does not exist"
            )

        if not directory.is_dir():
            return ValidationResult(valid=False, path=str(directory), reason="Path is not a directory")

        try:
            # Test read access
            list(directory.iterdir())
            return ValidationResult(valid=True, path=str(directory))
        except PermissionError:
            return ValidationResult(
                valid=False, path=str(directory), reason="Permission denied"
            )

    @staticmethod
    def validate_file(file_path: str | Path) -> ValidationResult:
        """Validate a file exists and is accessible.

        Args:
            file_path: File to validate

        Returns:
            Validation result
        """
        path = Path(file_path)

        if not path.exists():
            return ValidationResult(valid=False, path=str(path), reason="File does not exist")

        if not path.is_file():
            return ValidationResult(valid=False, path=str(path), reason="Path is not a file")

        try:
            # Test read access
            with open(path, "rb"):
                pass
            return ValidationResult(valid=True, path=str(path))
        except PermissionError:
            return ValidationResult(valid=False, path=str(path), reason="Permission denied")


class ConfigurationValidator:
    """Validates configuration paths and settings."""

    @staticmethod
    def validate_path_configuration() -> dict[str, Any]:
        """Validate configured paths.

        Returns:
            Path validation results
        """
        settings = get_settings()

        paths: dict[str, Any] = {
            "image_path": settings.core.image_path,
            "screenshot_path": settings.screenshot.path,
            "current_directory": os.getcwd(),
            "home_directory": str(Path.home()),
        }

        # Validate image path
        image_path = Path(settings.core.image_path)
        image_result = PathValidator.validate_directory(image_path)
        paths["image_path_valid"] = image_result.valid
        paths["image_path_reason"] = image_result.reason

        # Validate screenshot path
        screenshot_path = Path(settings.screenshot.path)
        screenshot_result = PathValidator.validate_directory(screenshot_path)
        paths["screenshot_path_valid"] = screenshot_result.valid
        paths["screenshot_path_reason"] = screenshot_result.reason

        return paths


class DirectoryAnalyzer:
    """Analyzes directory structure for image files."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

    @staticmethod
    def analyze_directory(directory: Path) -> dict[str, Any]:
        """Analyze a directory for image files.

        Args:
            directory: Directory to analyze

        Returns:
            Directory analysis with file counts and types
        """
        from collections import defaultdict

        file_types: defaultdict[str, int] = defaultdict(int)

        analysis: dict[str, Any] = {
            "exists": True,
            "path": str(directory),
            "total_files": 0,
            "image_files": 0,
            "subdirectories": 0,
            "file_types": file_types,
            "total_size_mb": 0.0,
        }

        for item in directory.rglob("*"):
            if item.is_file():
                analysis["total_files"] += 1
                extension = item.suffix.lower()
                file_types[extension] += 1

                if extension in DirectoryAnalyzer.IMAGE_EXTENSIONS:
                    analysis["image_files"] += 1

                try:
                    analysis["total_size_mb"] += item.stat().st_size / (1024 * 1024)
                except (OSError, PermissionError):
                    pass

            elif item.is_dir():
                analysis["subdirectories"] += 1

        analysis["file_types"] = dict(file_types)
        return analysis

    @staticmethod
    def analyze_configured_directories() -> dict[str, Any]:
        """Analyze all configured directories.

        Returns:
            Analysis results for image and screenshot directories
        """
        settings = get_settings()
        results = {}

        # Analyze image directory
        image_dir = Path(settings.core.image_path)
        if image_dir.exists():
            results["image_directory"] = DirectoryAnalyzer.analyze_directory(image_dir)
        else:
            results["image_directory"] = {"exists": False, "path": str(image_dir)}

        # Analyze screenshot directory
        screenshot_dir = Path(settings.screenshot.path)
        if screenshot_dir.exists():
            results["screenshot_directory"] = DirectoryAnalyzer.analyze_directory(screenshot_dir)
        else:
            results["screenshot_directory"] = {"exists": False, "path": str(screenshot_dir)}

        return results

    @staticmethod
    def find_all_images(directory: Path) -> list[str]:
        """Find all image files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of image file paths
        """
        if not directory.exists():
            return []

        images = []
        for item in directory.rglob("*"):
            if item.is_file() and item.suffix.lower() in DirectoryAnalyzer.IMAGE_EXTENSIONS:
                images.append(str(item))

        return images
