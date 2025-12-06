"""Manages image asset extraction and storage."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config_parser import QontinuiConfig

logger = logging.getLogger(__name__)


class ImageAssetManager:
    """Manages extraction, storage, and lifecycle of image assets.

    ImageAssetManager separates image management concerns from configuration parsing.
    It handles:
    - Building image lookup maps from configuration
    - Saving images to temporary files for OpenCV usage
    - Cleaning up temporary directories on completion

    The manager works with ImageExtractor to build a comprehensive image_map that
    includes both direct image assets and StateImage-derived images.

    Attributes:
        temp_dir: Temporary directory where images are saved for execution.

    Example:
        >>> manager = ImageAssetManager()
        >>> manager.save_images(config)
        >>> # ... use images during execution ...
        >>> manager.cleanup()  # Clean up temp files
    """

    def __init__(self) -> None:
        """Initialize image asset manager."""
        self.temp_dir: Path | None = None

    def save_images(self, config: QontinuiConfig) -> None:
        """Save all images to temporary files for OpenCV usage.

        Creates a temporary directory and saves all images from the config's
        image_map to disk. Updates each ImageAsset's file_path attribute.

        Args:
            config: Configuration with image_map to save.

        Note:
            - Creates temp directory with qontinui_images_ prefix
            - Only saves images that haven't been saved yet (file_path is None)
            - Sets config.image_directory for reference
        """
        # Create temporary directory for images
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qontinui_images_"))
        config.image_directory = self.temp_dir

        # Save all images from image_map
        saved_count = 0
        for _image_id, image in config.image_map.items():
            try:
                if image.file_path is None:  # Only save if not already saved
                    image.save_to_file(self.temp_dir)
                    logger.debug(f"Saved image: {image.name} to {image.file_path}")
                    saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save image {image.name}: {e}")

        logger.debug(f"Saved {saved_count} images to {self.temp_dir}")

    def cleanup(self) -> None:
        """Clean up temporary files and directory.

        Removes the temporary directory and all saved images.
        Should be called when automation execution completes.
        """
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
