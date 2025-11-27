"""SimilarityCalculator for comparing images with mask support.

This class calculates similarity between two images, taking into account
optional masks that define which pixels should be considered in the comparison.
Extracted from Pattern class following Single Responsibility Principle.
"""

from typing import Any

import numpy as np


class SimilarityCalculator:
    """
    Calculate similarity between images with mask support.

    This calculator compares two images pixel-by-pixel, considering optional
    masks that define active regions. The similarity score ranges from 0.0
    (completely different) to 1.0 (identical).

    The calculation:
    1. Validates that images have compatible shapes
    2. Applies masks to both images (only active pixels are compared)
    3. Calculates normalized pixel differences
    4. Returns weighted similarity score

    Constants:
        PIXEL_MAX_VALUE: Maximum pixel value for normalization (255.0)
        MASK_ACTIVE_THRESHOLD: Threshold for considering a mask pixel active (0.5)
    """

    PIXEL_MAX_VALUE = 255.0
    MASK_ACTIVE_THRESHOLD = 0.5

    def calculate_similarity(
        self,
        img1: np.ndarray[Any, Any],
        mask1: np.ndarray[Any, Any],
        img2: np.ndarray[Any, Any],
        mask2: np.ndarray[Any, Any] | None = None,
    ) -> float:
        """
        Calculate similarity between two images with mask support.

        Compares two images pixel-by-pixel, considering only the pixels
        marked as active in both masks. Returns a similarity score between
        0.0 (completely different) and 1.0 (identical).

        Args:
            img1: First image array (H x W x C)
            mask1: Mask for first image (H x W), values 0.0-1.0
            img2: Second image array (H x W x C)
            mask2: Optional mask for second image (H x W), values 0.0-1.0.
                   If None, uses full mask (all pixels active)

        Returns:
            Similarity score (0.0-1.0)

        Notes:
            - Images must have identical shapes
            - Masks define which pixels are compared (0.0=ignore, 1.0=full weight)
            - Only pixels active in BOTH masks are compared
            - Returns 0.0 if no pixels are active in combined mask
        """
        # Validate shapes
        if not self._validate_shapes(img1, img2):
            return 0.0

        # Create full mask if not provided
        if mask2 is None:
            mask2 = np.ones(img2.shape[:2], dtype=np.float32)

        # Apply masks and calculate difference
        masked_img1, masked_img2, combined_mask = self._apply_masks(img1, mask1, img2, mask2)

        # Calculate weighted difference
        return self._calculate_weighted_difference(masked_img1, masked_img2, combined_mask)

    def _validate_shapes(self, img1: np.ndarray[Any, Any], img2: np.ndarray[Any, Any]) -> bool:
        """
        Validate that two images have compatible shapes.

        Args:
            img1: First image array
            img2: Second image array

        Returns:
            True if shapes are compatible, False otherwise
        """
        return img1.shape == img2.shape

    def _apply_masks(
        self,
        img1: np.ndarray[Any, Any],
        mask1: np.ndarray[Any, Any],
        img2: np.ndarray[Any, Any],
        mask2: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Apply masks to images and create combined mask.

        Converts images to float32 for precise calculation and applies
        masks to both images. Creates a combined mask where only pixels
        active in both masks are considered.

        Args:
            img1: First image array
            mask1: Mask for first image
            img2: Second image array
            mask2: Mask for second image

        Returns:
            Tuple of (masked_img1, masked_img2, combined_mask)
        """
        # Convert to float for precise calculation
        float_img1 = img1.astype(np.float32)
        float_img2 = img2.astype(np.float32)

        # Expand masks to match image dimensions
        if len(float_img1.shape) == 3:
            # Color image (H x W x C) - expand masks to (H x W x C)
            mask1_expanded = np.expand_dims(mask1, axis=2)
            mask2_expanded = np.expand_dims(mask2, axis=2)
        else:
            # Grayscale image (H x W) - use masks as-is
            mask1_expanded = mask1
            mask2_expanded = mask2

        # Apply masks to images
        masked_img1 = float_img1 * mask1_expanded
        masked_img2 = float_img2 * mask2_expanded

        # Combined mask (both pixels must be active)
        combined_mask = mask1_expanded * mask2_expanded

        return masked_img1, masked_img2, combined_mask

    def _calculate_weighted_difference(
        self,
        masked_img1: np.ndarray[Any, Any],
        masked_img2: np.ndarray[Any, Any],
        combined_mask: np.ndarray[Any, Any],
    ) -> float:
        """
        Calculate weighted difference between masked images.

        Computes the normalized difference between two masked images,
        considering only pixels marked as active in the combined mask.

        Args:
            masked_img1: First masked image
            masked_img2: Second masked image
            combined_mask: Combined mask (active pixels in both masks)

        Returns:
            Similarity score (0.0-1.0)
        """
        # Calculate absolute difference
        diff = np.abs(masked_img1 - masked_img2)

        # Normalize difference by pixel max value
        normalized_diff = diff / self.PIXEL_MAX_VALUE

        # Count active pixels (where mask > threshold)
        active_pixels = np.sum(combined_mask > self.MASK_ACTIVE_THRESHOLD)

        # No active pixels means no similarity
        if active_pixels == 0:
            return 0.0

        # Calculate weighted average difference
        weighted_diff_sum = np.sum(normalized_diff * combined_mask)
        avg_diff = weighted_diff_sum / active_pixels

        # Convert difference to similarity (1.0 = identical, 0.0 = completely different)
        similarity = 1.0 - avg_diff

        return float(similarity)
