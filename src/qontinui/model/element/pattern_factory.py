"""Factory for creating Pattern instances from various sources.

This module contains all factory methods for Pattern creation,
separated from the Pattern class to follow the Single Responsibility Principle.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
from qontinui_schemas.common import utc_now

if TYPE_CHECKING:
    from .pattern import Pattern


class PatternFactory:
    """Factory class for creating Pattern instances from various sources.

    All methods are static as this is a stateless factory.
    Supports creation from images, files, matches, and state images.
    """

    @staticmethod
    def from_image(image: Any, name: str | None = None, pattern_id: str | None = None) -> Pattern:
        """Create Pattern from Image with full mask.

        Args:
            image: Image object or numpy array
            name: Optional pattern name
            pattern_id: Optional pattern ID

        Returns:
            Pattern instance with full mask

        Raises:
            ValueError: If image type is invalid or pixel data cannot be extracted
        """
        from .image import Image
        from .pattern import Pattern

        # Convert to numpy array
        if isinstance(image, Image):
            pixel_data = image.get_mat_bgr()
            if name is None:
                name = image.name
        elif isinstance(image, np.ndarray):
            pixel_data = image
        else:
            raise ValueError(f"Invalid image type: {type(image)}")

        if pixel_data is None:
            raise ValueError("Could not extract pixel data from image")

        # Create full mask (all pixels active)
        mask = np.ones(pixel_data.shape[:2], dtype=np.float32)

        if pattern_id is None:
            pattern_id = PatternFactory._generate_pattern_id(pixel_data)

        return Pattern(
            id=pattern_id,
            name=name or "pattern",
            pixel_data=pixel_data,
            mask=mask,
            width=pixel_data.shape[1],
            height=pixel_data.shape[0],
        )

    @staticmethod
    def from_file(img_path: str, name: str | None = None) -> Pattern:
        """Create Pattern from image file.

        Args:
            img_path: Path to image file
            name: Optional pattern name (defaults to filename stem)

        Returns:
            Pattern instance

        Raises:
            ValueError: If file cannot be loaded or is invalid
        """
        from pathlib import Path

        from .image import Image

        image = Image.from_file(img_path)
        if name is None:
            name = Path(img_path).stem

        return PatternFactory.from_image(image, name=name)

    @staticmethod
    def from_match(match: Any, pattern_id: str | None = None) -> Pattern:
        """Create Pattern from a Match object.

        Args:
            match: Match object to create pattern from
            pattern_id: Optional custom ID

        Returns:
            Pattern instance

        Raises:
            ValueError: If match object is invalid
        """
        from .pattern import Pattern

        if pattern_id is None:
            pattern_id = f"pattern_{hashlib.md5(str(match).encode(), usedforsecurity=False).hexdigest()[:8]}"

        # Extract pixel data from match
        pixel_data = (
            match.image.get_mat_bgr() if match.image else np.zeros((10, 10, 3), dtype=np.uint8)
        )

        # Create full mask
        mask = np.ones(pixel_data.shape[:2], dtype=np.float32)

        region = match.get_region()
        return Pattern(
            id=pattern_id,
            name=match.name or "pattern_from_match",
            pixel_data=pixel_data,
            mask=mask,
            x=region.x if region else 0,
            y=region.y if region else 0,
            width=region.width if region else pixel_data.shape[1],
            height=region.height if region else pixel_data.shape[0],
            similarity=match.score if hasattr(match, "score") else 0.95,
        )

    @staticmethod
    def from_state_image(state_image: Any, pattern_id: str | None = None) -> Pattern:
        """Create a Pattern from a StateImage.

        Args:
            state_image: StateImage object
            pattern_id: Optional custom ID

        Returns:
            Pattern instance

        Raises:
            ValueError: If state_image is invalid
        """
        from .pattern import Pattern

        if pattern_id is None:
            pattern_id = f"pattern_{state_image.id}"

        # Extract pixel data and mask
        pixel_data = state_image.pixel_data
        mask = state_image.mask if state_image.mask is not None else np.ones(pixel_data.shape[:2])

        return Pattern(
            id=pattern_id,
            name=state_image.name,
            pixel_data=pixel_data,
            mask=mask,
            x=state_image.x,
            y=state_image.y,
            width=state_image.x2 - state_image.x,
            height=state_image.y2 - state_image.y,
            mask_density=(
                state_image.mask_density if hasattr(state_image, "mask_density") else 1.0
            ),
            mask_type="imported",
            tags=state_image.tags if hasattr(state_image, "tags") else [],
            created_at=(
                state_image.created_at if hasattr(state_image, "created_at") else utc_now()
            ),
        )

    @staticmethod
    def _generate_pattern_id(pixel_data: np.ndarray[Any, Any]) -> str:
        """Generate a unique pattern ID from pixel data.

        Args:
            pixel_data: Image pixel data

        Returns:
            Unique pattern ID string
        """
        return f"pattern_{hashlib.md5(pixel_data.tobytes(), usedforsecurity=False).hexdigest()[:8]}"
