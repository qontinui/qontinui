"""Pattern mask optimization algorithms.

This module provides mask optimization capabilities for Pattern objects.
Extracted from Pattern class to follow Single Responsibility Principle.

Optimization Methods:
- Stability-based: Find stable pixels across multiple samples
- Discriminative: Maximize discrimination between positive/negative samples
"""

from datetime import datetime
from typing import Any

import numpy as np


class PatternOptimizer:
    """Handles mask optimization for Pattern objects.

    This class implements various algorithms for optimizing pattern masks
    based on positive and negative sample images. The goal is to identify
    which pixels are most important for pattern matching.

    Optimization Methods:
        - stability: Identifies stable pixels across positive samples
        - discriminative: Maximizes discrimination between positive/negative samples

    Example:
        >>> optimizer = PatternOptimizer()
        >>> mask, metrics = optimizer.optimize_mask(
        ...     pattern.pixel_data,
        ...     pattern.mask,
        ...     positive_samples=[img1, img2],
        ...     method="stability"
        ... )
        >>> pattern.mask = mask
    """

    # Configuration constants
    STABILITY_THRESHOLD = 0.7  # Threshold for stable pixel detection
    DISCRIMINATION_WEIGHT = 0.5  # Weight for discrimination vs stability

    def optimize_mask(
        self,
        pixel_data: np.ndarray[Any, Any],
        current_mask: np.ndarray[Any, Any],
        positive_samples: list[np.ndarray[Any, Any]],
        negative_samples: list[np.ndarray[Any, Any]] | None = None,
        method: str = "stability",
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Optimize mask based on positive and negative samples.

        Args:
            pixel_data: Original pattern image data (H x W x C)
            current_mask: Current mask array (H x W)
            positive_samples: Images that should match
            negative_samples: Images that should not match
            method: Optimization method ("stability" or "discriminative")

        Returns:
            Tuple of (optimized mask, optimization metrics)

        Raises:
            ValueError: If method is unknown
        """
        if method == "stability":
            return self._optimize_by_stability(pixel_data, positive_samples)
        elif method == "discriminative":
            return self._optimize_discriminative(pixel_data, positive_samples, negative_samples)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _optimize_by_stability(
        self,
        pixel_data: np.ndarray[Any, Any],
        positive_samples: list[np.ndarray[Any, Any]],
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Optimize mask by finding stable pixels across positive samples.

        Stable pixels are those with low variance across multiple screenshots
        of the same object. These are the most reliable for pattern matching.

        Args:
            pixel_data: Original pattern image
            positive_samples: List of positive sample images

        Returns:
            Tuple of (optimized mask, metrics dict)
        """
        if not positive_samples:
            mask_shape = pixel_data.shape[:2]
            return np.ones(mask_shape, dtype=np.float32), {"error": "No positive samples provided"}

        # Stack all samples including original
        all_samples = np.stack([pixel_data] + positive_samples, axis=0)

        # Calculate pixel variance
        if len(all_samples.shape) == 4:  # Color images
            # Convert to grayscale for stability calculation
            gray_samples = np.mean(all_samples, axis=3)
        else:
            gray_samples = all_samples

        # Calculate standard deviation for each pixel
        pixel_std = np.std(gray_samples, axis=0)

        # Create stability mask (low variance = stable = important)
        max_std = np.max(pixel_std)
        if max_std > 0:
            stability = 1.0 - (pixel_std / max_std)
        else:
            stability = np.ones_like(pixel_std)

        # Threshold to create binary mask
        optimized_mask = np.where(stability >= self.STABILITY_THRESHOLD, 1.0, 0.0).astype(
            np.float32
        )

        # Calculate metrics
        metrics = {
            "method": "stability",
            "samples_used": len(positive_samples),
            "avg_stability": float(np.mean(stability)),
            "mask_density": float(np.sum(optimized_mask > 0.5) / optimized_mask.size),
            "threshold": self.STABILITY_THRESHOLD,
        }

        return optimized_mask, metrics

    def _optimize_discriminative(
        self,
        pixel_data: np.ndarray[Any, Any],
        positive_samples: list[np.ndarray[Any, Any]],
        negative_samples: list[np.ndarray[Any, Any]] | None = None,
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Optimize mask to maximize discrimination between positive and negative samples.

        This method combines stability analysis with discrimination analysis.
        It identifies pixels that not only are stable within positive samples,
        but also differ significantly from negative samples.

        Args:
            pixel_data: Original pattern image
            positive_samples: Images that should match
            negative_samples: Images that should not match

        Returns:
            Tuple of (optimized mask, metrics dict)
        """
        # Start with stability-based mask
        stability_mask, _ = self._optimize_by_stability(pixel_data, positive_samples)

        if not negative_samples:
            return stability_mask, {"method": "discriminative_stability_only"}

        # For each pixel, calculate discrimination power
        h, w = pixel_data.shape[:2]
        discrimination = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # Skip if pixel is not stable
                if stability_mask[y, x] < 0.5:
                    continue

                # Get pixel values from positive samples
                pos_values = [img[y, x] for img in positive_samples]
                neg_values = [img[y, x] for img in negative_samples if img.shape[:2] == (h, w)]

                if not neg_values:
                    discrimination[y, x] = stability_mask[y, x]
                    continue

                # Calculate separation between positive and negative
                pos_mean = np.mean(pos_values, axis=0)
                neg_mean = np.mean(neg_values, axis=0)

                # Distance between means
                if len(pos_mean.shape) > 0:  # Color pixel
                    dist = np.linalg.norm(pos_mean - neg_mean)
                else:  # Grayscale pixel
                    dist = abs(pos_mean - neg_mean)

                discrimination[y, x] = min(dist / 255.0, 1.0)

        # Combine stability and discrimination
        optimized_mask = (
            stability_mask * self.DISCRIMINATION_WEIGHT
            + discrimination * self.DISCRIMINATION_WEIGHT
        )

        # Threshold
        optimized_mask = np.where(optimized_mask >= 0.5, 1.0, 0.0).astype(np.float32)

        metrics = {
            "method": "discriminative",
            "positive_samples": len(positive_samples),
            "negative_samples": len(negative_samples) if negative_samples else 0,
            "mask_density": float(np.sum(optimized_mask > 0.5) / optimized_mask.size),
        }

        return optimized_mask, metrics

    def update_mask(
        self,
        pattern: Any,
        new_mask: np.ndarray[Any, Any],
        record_history: bool = True,
    ) -> None:
        """Update pattern's mask with optimization tracking.

        Args:
            pattern: Pattern object to update
            new_mask: New mask array
            record_history: Whether to record this update in optimization history

        Raises:
            ValueError: If new mask shape doesn't match current mask
        """
        if new_mask.shape != pattern.mask.shape:
            raise ValueError(
                f"New mask shape {new_mask.shape} doesn't match "
                f"current shape {pattern.mask.shape}"
            )

        old_density = pattern.mask_density

        pattern.mask = new_mask
        pattern.mask_density = float(np.sum(new_mask > 0.5) / new_mask.size)
        pattern.updated_at = datetime.now()

        if record_history:
            pattern.optimization_history.append(
                {
                    "timestamp": pattern.updated_at.isoformat(),
                    "old_density": old_density,
                    "new_density": pattern.mask_density,
                    "operation": "mask_update",
                }
            )

    def add_variation(self, pattern: Any, image: np.ndarray[Any, Any]) -> None:
        """Add image variation to pattern for optimization.

        Args:
            pattern: Pattern object to update
            image: Image variation (same object, different screenshot)

        Raises:
            ValueError: If variation shape doesn't match pattern shape
        """
        if image.shape != pattern.pixel_data.shape:
            raise ValueError("Variation shape doesn't match pattern shape")

        pattern.variations.append(image)
