"""Image loading operations for diagnostics.

Handles actual image loading with timing and error tracking.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from .image_validator import PathValidator


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


class ImageLoader:
    """Loads images with timing and error tracking."""

    def __init__(self) -> None:
        """Initialize the image loader."""
        self.load_history: dict[str, LoadResult] = {}

    def load_image(self, image_name: str) -> LoadResult:
        """Load a single image.

        Args:
            image_name: Name or path of the image to load

        Returns:
            Load result with timing and status
        """
        start_time = time.time()

        try:
            from PIL import Image

            # Get candidate paths
            paths_to_try = PathValidator.validate_image_path(image_name)

            # Try each path
            for path in paths_to_try:
                try:
                    img = Image.open(path)
                    load_time_ms = (time.time() - start_time) * 1000

                    result = LoadResult(
                        success=True,
                        image_name=image_name,
                        loaded_from=str(path),
                        load_time_ms=load_time_ms,
                        image_dimensions=img.size,
                    )
                    self.load_history[image_name] = result
                    return result

                except (OSError, FileNotFoundError, ValueError):
                    continue

            # All paths failed
            result = LoadResult(
                success=False,
                image_name=image_name,
                failure_reason="Image not found in any configured path",
            )
            self.load_history[image_name] = result
            return result

        except (ImportError, AttributeError) as e:
            result = LoadResult(success=False, image_name=image_name, failure_reason=str(e))
            self.load_history[image_name] = result
            return result

    def load_multiple_images(self, image_names: list[str]) -> list[LoadResult]:
        """Load multiple images.

        Args:
            image_names: List of image names to load

        Returns:
            List of load results
        """
        return [self.load_image(name) for name in image_names]

    def load_all_images_in_directory(self, directory: Path) -> dict[str, int | float]:
        """Load all images in a directory.

        Args:
            directory: Directory containing images

        Returns:
            Summary statistics
        """
        from .image_validator import DirectoryAnalyzer

        all_images = DirectoryAnalyzer.find_all_images(directory)

        if not all_images:
            return {
                "total_images": 0,
                "successful": 0,
                "failed": 0,
                "average_load_time_ms": 0.0,
                "success_rate": 0.0,
            }

        successful = 0
        failed = 0
        total_time = 0.0

        for image_path in all_images:
            result = self.load_image(image_path)
            if result.success:
                successful += 1
                if result.load_time_ms:
                    total_time += result.load_time_ms
            else:
                failed += 1

        return {
            "total_images": len(all_images),
            "successful": successful,
            "failed": failed,
            "average_load_time_ms": total_time / successful if successful > 0 else 0.0,
            "success_rate": (successful / len(all_images) * 100) if all_images else 0.0,
        }

    def get_load_statistics(self) -> dict[str, int | float]:
        """Get statistics from load history.

        Returns:
            Load statistics including success rate and timing
        """
        if not self.load_history:
            return {"message": "No images loaded yet"}

        total_attempts = len(self.load_history)
        successful = sum(1 for r in self.load_history.values() if r.success)
        failed = total_attempts - successful

        load_times = [
            r.load_time_ms for r in self.load_history.values() if r.load_time_ms is not None
        ]

        stats: dict[str, int | float] = {
            "total_attempts": total_attempts,
            "successful_loads": successful,
            "failed_loads": failed,
            "success_rate": (successful / total_attempts * 100) if total_attempts > 0 else 0.0,
        }

        if load_times:
            stats["average_load_time_ms"] = sum(load_times) / len(load_times)
            stats["min_load_time_ms"] = min(load_times)
            stats["max_load_time_ms"] = max(load_times)

        return stats

    def clear_history(self) -> None:
        """Clear the load history."""
        self.load_history.clear()

    def get_failed_loads(self) -> list[LoadResult]:
        """Get all failed load attempts.

        Returns:
            List of failed load results
        """
        return [result for result in self.load_history.values() if not result.success]

    def get_successful_loads(self) -> list[LoadResult]:
        """Get all successful load attempts.

        Returns:
            List of successful load results
        """
        return [result for result in self.load_history.values() if result.success]
