"""Extracts and processes image assets from configuration."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config_parser import ImageAsset, Pattern, QontinuiConfig, State, StateImage

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extracts and processes image assets from configuration.

    ImageExtractor separates image processing logic from configuration parsing.
    Patterns reference images via imageId from the config.images array:
    - Pattern has imageId field pointing to existing ImageAsset
    - StateImage.id becomes an alias to the referenced ImageAsset

    The extractor builds image_map that allows both direct image references
    and StateImage ID lookups.

    Example:
        >>> extractor = ImageExtractor()
        >>> image_map = extractor.extract_images(config)
        >>> # Access image by StateImage ID
        >>> button_image = image_map["login_button_state_img"]
        >>> # Or by direct image ID
        >>> logo_image = image_map["logo_img_001"]
    """

    def extract_images(self, config: "QontinuiConfig") -> dict[str, "ImageAsset"]:
        """Extract all images from configuration and build image_map.

        Processes both direct image assets and StateImage patterns to create
        a unified image_map that supports lookups by image ID or StateImage ID.

        Args:
            config: QontinuiConfig object with images and states to process.

        Returns:
            Dictionary mapping image IDs and StateImage IDs to ImageAsset objects.
            Keys include both config.images IDs and state.identifying_images IDs.

        Example:
            >>> image_map = extractor.extract_images(config)
            >>> print(f"Extracted {len(image_map)} images")
            Extracted 42 images
        """
        # Start with images from config.images array
        image_map: dict[str, ImageAsset] = {img.id: img for img in config.images}

        # Extract images from StateImage patterns
        stateimage_assets = self._extract_from_state_images(config.states, image_map)
        image_map.update(stateimage_assets)

        stateimage_count = len(stateimage_assets)
        logger.debug(
            f"image_map now contains {len(image_map)} entries "
            f"({stateimage_count} StateImages added)"
        )
        logger.debug(f"image_map keys: {list(image_map.keys())}")

        return image_map

    def _extract_from_state_images(
        self, states: list["State"], existing_image_map: dict[str, "ImageAsset"]
    ) -> dict[str, "ImageAsset"]:
        """Extract images from StateImage patterns in all states.

        Processes each state's identifying images and creates ImageAsset objects
        from their patterns using imageId references.

        Args:
            states: List of State objects containing identifying_images.
            existing_image_map: Dictionary of already loaded images from config.images.

        Returns:
            Dictionary mapping StateImage IDs to ImageAsset objects.

        Note:
            Currently processes only the first pattern of each StateImage.
            Future enhancement: handle multiple patterns per StateImage.
        """
        image_assets: dict[str, ImageAsset] = {}

        for state in states:
            for state_image in state.identifying_images:
                if not state_image.patterns:
                    logger.warning(f"StateImage {state_image.id} has no patterns")
                    continue

                # Use the first pattern's image data
                # Future enhancement: handle multiple patterns per StateImage
                pattern = state_image.patterns[0]

                asset = self._create_from_pattern(state_image, pattern, existing_image_map)
                if asset:
                    image_assets[state_image.id] = asset

        return image_assets

    def _create_from_pattern(
        self,
        state_image: "StateImage",
        pattern: "Pattern",
        existing_image_map: dict[str, "ImageAsset"],
    ) -> "ImageAsset | None":
        """Create ImageAsset from StateImage pattern data.

        Pattern must have image_id referencing an existing image in config.images.

        Args:
            state_image: StateImage containing the pattern.
            pattern: Pattern object with image reference.
            existing_image_map: Dictionary of already loaded images from config.images.

        Returns:
            ImageAsset if creation succeeds, None if pattern has no image reference.

        Note:
            Returns the existing ImageAsset that the pattern references.
        """
        # Pattern must have imageId reference
        if pattern.image_id:
            # Pattern references an existing image in the images array
            if pattern.image_id in existing_image_map:
                logger.debug(f"StateImage {state_image.id} -> references image {pattern.image_id}")
                return existing_image_map[pattern.image_id]
            else:
                logger.warning(
                    f"StateImage {state_image.id} references missing image {pattern.image_id}"
                )
                return None

        # No image reference found
        logger.warning(f"StateImage {state_image.id} has no image_id reference")
        return None
