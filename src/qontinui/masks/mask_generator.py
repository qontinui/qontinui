"""Mask generation utilities for pattern matching."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import cv2
import numpy as np


class MaskType(Enum):
    """Types of masks that can be generated."""

    FULL = "full"  # Full rectangular mask (all pixels active)
    STABILITY = "stability"  # Based on pixel stability across screenshots
    EDGE = "edge"  # Based on edge detection
    SALIENCY = "saliency"  # Based on visual saliency
    TRANSPARENCY = "transparency"  # Based on alpha channel
    MANUAL = "manual"  # User-defined mask
    ADAPTIVE = "adaptive"  # Combination of multiple techniques


@dataclass
class MaskMetadata:
    """Metadata about a generated mask."""

    type: MaskType
    density: float  # Percentage of active pixels (0.0-1.0)
    active_pixels: int
    total_pixels: int
    generation_params: dict[str, Any]


class MaskGenerator:
    """Generate masks for image regions using various techniques."""

    def __init__(self) -> None:
        """Initialize the mask generator."""
        pass

    def generate_mask(
        self, image: np.ndarray[Any, Any], mask_type: MaskType = MaskType.FULL, **kwargs
    ) -> tuple[np.ndarray[Any, Any], MaskMetadata]:
        """
        Generate a mask for the given image.

        Args:
            image: Input image (H x W x C)
            mask_type: Type of mask to generate
            **kwargs: Additional parameters for specific mask types

        Returns:
            Tuple of (mask array, metadata)
        """
        if mask_type == MaskType.FULL:
            h, w = image.shape[:2]
            return self.generate_full_mask((h, w))
        elif mask_type == MaskType.STABILITY:
            return self.generate_stability_mask(image, **kwargs)
        elif mask_type == MaskType.EDGE:
            return self.generate_edge_mask(image, **kwargs)
        elif mask_type == MaskType.SALIENCY:
            return self.generate_saliency_mask(image, **kwargs)
        elif mask_type == MaskType.TRANSPARENCY:
            return self.generate_transparency_mask(image, **kwargs)
        elif mask_type == MaskType.ADAPTIVE:
            return self.generate_adaptive_mask(image, **kwargs)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    def generate_full_mask(
        self, shape: tuple[int, int]
    ) -> tuple[np.ndarray[Any, Any], MaskMetadata]:
        """
        Generate a full rectangular mask with all pixels active.

        Args:
            shape: (height, width) of the mask

        Returns:
            Tuple of (mask array, metadata)
        """
        mask = np.ones(shape, dtype=np.float32)

        metadata = MaskMetadata(
            type=MaskType.FULL,
            density=1.0,
            active_pixels=int(np.prod(shape)),
            total_pixels=int(np.prod(shape)),
            generation_params={},
        )

        return mask, metadata

    def generate_stability_mask(
        self,
        image: np.ndarray[Any, Any],
        variations: list[np.ndarray[Any, Any]] | None = None,
        stability_threshold: float = 0.95,
        **kwargs,
    ) -> tuple[np.ndarray[Any, Any], MaskMetadata]:
        """
        Generate a mask based on pixel stability across variations.

        Args:
            image: Reference image
            variations: List of image variations (same region from different screenshots)
            stability_threshold: Threshold for considering a pixel stable (0.0-1.0)

        Returns:
            Tuple of (mask array, metadata)
        """
        height, width = image.shape[:2]

        if variations is None or len(variations) == 0:
            # No variations provided, return full mask
            return self.generate_full_mask((height, width))

        # Calculate pixel variance across variations
        all_images = np.stack([image] + variations, axis=0)

        # Convert to grayscale for stability calculation
        if len(all_images.shape) == 4 and all_images.shape[3] == 3:
            gray_images = np.mean(all_images, axis=3)
        else:
            gray_images = all_images

        # Calculate standard deviation for each pixel
        pixel_std = np.std(gray_images, axis=0)

        # Normalize standard deviation to 0-1 range
        max_std = np.max(pixel_std)
        if max_std > 0:
            normalized_std = pixel_std / max_std
        else:
            normalized_std = np.zeros_like(pixel_std)

        # Create mask: stable pixels (low variance) are active
        mask = (1.0 - normalized_std).astype(np.float32)
        mask = np.where(mask >= stability_threshold, 1.0, mask)

        active_pixels = np.sum(mask > 0.5)
        metadata = MaskMetadata(
            type=MaskType.STABILITY,
            density=float(active_pixels / mask.size),
            active_pixels=int(active_pixels),
            total_pixels=int(mask.size),
            generation_params={"stability_threshold": stability_threshold},
        )

        return mask, metadata

    def generate_edge_mask(
        self,
        image: np.ndarray[Any, Any],
        low_threshold: int = 50,
        high_threshold: int = 150,
        dilation_size: int = 3,
        **kwargs,
    ) -> tuple[np.ndarray[Any, Any], MaskMetadata]:
        """
        Generate a mask based on edge detection.

        Args:
            image: Input image
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection
            dilation_size: Size of dilation kernel to expand edges

        Returns:
            Tuple of (mask array, metadata)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Dilate edges to create regions
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Convert to float mask
        mask = dilated.astype(np.float32) / 255.0

        # Add some gradient around edges for smoother matching
        mask = cv2.GaussianBlur(mask, (5, 5), 1.0).astype(np.float32)

        active_pixels = np.sum(mask > 0.5)
        metadata = MaskMetadata(
            type=MaskType.EDGE,
            density=float(active_pixels / mask.size),
            active_pixels=int(active_pixels),
            total_pixels=int(mask.size),
            generation_params={
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "dilation_size": dilation_size,
            },
        )

        return mask, metadata

    def generate_saliency_mask(
        self,
        image: np.ndarray[Any, Any],
        method: str = "spectral_residual",
        threshold: float = 0.5,
        **kwargs,
    ) -> tuple[np.ndarray[Any, Any], MaskMetadata]:
        """
        Generate a mask based on visual saliency.

        Args:
            image: Input image
            method: Saliency detection method
            threshold: Threshold for saliency map

        Returns:
            Tuple of (mask array, metadata)
        """
        if method == "spectral_residual":
            mask = self._spectral_residual_saliency(image)
        else:
            # Simple center-weighted saliency as fallback
            h, w = image.shape[:2]
            mask = self._center_weighted_saliency((h, w))

        # Apply threshold
        mask = np.where(mask >= threshold, 1.0, mask).astype(np.float32)

        active_pixels = np.sum(mask > 0.5)
        metadata = MaskMetadata(
            type=MaskType.SALIENCY,
            density=float(active_pixels / mask.size),
            active_pixels=int(active_pixels),
            total_pixels=int(mask.size),
            generation_params={"method": method, "threshold": threshold},
        )

        return mask, metadata

    def _spectral_residual_saliency(
        self, image: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Compute saliency using spectral residual method.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Compute FFT
        img_float = gray.astype(np.float32) / 255.0
        fft = np.fft.fft2(img_float)

        # Compute log amplitude and phase
        log_amplitude = np.log(np.abs(fft) + 1e-7)
        phase = np.angle(fft)

        # Compute spectral residual
        spectral_residual = log_amplitude - cv2.boxFilter(log_amplitude, -1, (3, 3))

        # Reconstruct image
        combined = np.exp(spectral_residual + 1j * phase)
        img_back = np.fft.ifft2(combined)
        saliency = np.abs(img_back) ** 2

        # Smooth and normalize
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (11, 11), 2.5).astype(
            np.float32
        )
        saliency = (saliency - saliency.min()) / (
            saliency.max() - saliency.min() + 1e-7
        )

        return cast(np.ndarray[Any, Any], saliency)

    def _center_weighted_saliency(self, shape: tuple[int, int]) -> np.ndarray[Any, Any]:
        """
        Simple center-weighted saliency (objects tend to be in center).
        """
        h, w = shape
        y, x = np.ogrid[:h, :w]

        # Create Gaussian centered on image
        center_y, center_x = h / 2, w / 2
        sigma_y, sigma_x = h / 3, w / 3

        mask = np.exp(
            -(
                (x - center_x) ** 2 / (2 * sigma_x**2)
                + (y - center_y) ** 2 / (2 * sigma_y**2)
            )
        )

        return cast(np.ndarray[Any, Any], mask.astype(np.float32))

    def generate_transparency_mask(
        self, image: np.ndarray[Any, Any], alpha_threshold: float = 0.5, **kwargs
    ) -> tuple[np.ndarray[Any, Any], MaskMetadata]:
        """
        Generate a mask from the alpha channel of an image.

        Args:
            image: Input image with alpha channel (H x W x 4)
            alpha_threshold: Threshold for alpha channel (0.0-1.0)

        Returns:
            Tuple of (mask array, metadata)
        """
        if image.shape[2] < 4:
            # No alpha channel, return full mask
            h, w = image.shape[:2]
            return self.generate_full_mask((h, w))

        # Extract alpha channel and normalize to 0-1
        alpha = image[:, :, 3].astype(np.float32) / 255.0

        # Apply threshold
        mask = np.where(alpha >= alpha_threshold, 1.0, 0.0).astype(np.float32)

        active_pixels = np.sum(mask > 0.5)
        metadata = MaskMetadata(
            type=MaskType.TRANSPARENCY,
            density=float(active_pixels / mask.size),
            active_pixels=int(active_pixels),
            total_pixels=int(mask.size),
            generation_params={"alpha_threshold": alpha_threshold},
        )

        return mask, metadata

    def generate_adaptive_mask(
        self,
        image: np.ndarray[Any, Any],
        variations: list[np.ndarray[Any, Any]] | None = None,
        **kwargs,
    ) -> tuple[np.ndarray[Any, Any], MaskMetadata]:
        """
        Generate an adaptive mask combining multiple techniques.

        Args:
            image: Input image
            variations: Optional list of image variations

        Returns:
            Tuple of (mask array, metadata)
        """
        masks = []
        weights = []

        # Edge mask (important for UI elements)
        edge_mask, _ = self.generate_edge_mask(image)
        masks.append(edge_mask)
        weights.append(0.3)

        # Stability mask if variations provided
        if variations:
            stability_mask, _ = self.generate_stability_mask(image, variations)
            masks.append(stability_mask)
            weights.append(0.5)

        # Saliency mask
        saliency_mask, _ = self.generate_saliency_mask(image)
        masks.append(saliency_mask)
        weights.append(0.2)

        # Combine masks with weights
        weights = np.array(weights) / np.sum(weights)  # Normalize weights
        combined_mask = np.zeros_like(masks[0])

        for mask, weight in zip(masks, weights, strict=False):
            combined_mask += mask * weight

        # Threshold and smooth
        combined_mask = np.where(combined_mask >= 0.5, 1.0, combined_mask)
        combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0.5)

        active_pixels = np.sum(combined_mask > 0.5)
        metadata = MaskMetadata(
            type=MaskType.ADAPTIVE,
            density=float(active_pixels / combined_mask.size),
            active_pixels=int(active_pixels),
            total_pixels=int(combined_mask.size),
            generation_params={"weights": weights.tolist()},
        )

        return combined_mask.astype(np.float32), metadata

    def refine_mask(
        self,
        mask: np.ndarray[Any, Any],
        operation: str,
        strength: float = 1.0,
        **kwargs,
    ) -> np.ndarray[Any, Any]:
        """
        Refine an existing mask.

        Args:
            mask: Input mask
            operation: Type of refinement ('erode', 'dilate', 'smooth', 'threshold')
            strength: Strength of the operation

        Returns:
            Refined mask
        """
        if operation == "erode":
            kernel_size = int(3 + 2 * strength)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.erode(mask, kernel, iterations=1)

        elif operation == "dilate":
            kernel_size = int(3 + 2 * strength)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.dilate(mask, kernel, iterations=1)

        elif operation == "smooth":
            kernel_size = int(3 + 2 * strength) * 2 + 1  # Ensure odd
            return cv2.GaussianBlur(mask, (kernel_size, kernel_size), strength)

        elif operation == "threshold":
            threshold = kwargs.get("threshold", 0.5)
            return np.where(mask >= threshold, 1.0, 0.0).astype(np.float32)

        else:
            return mask

    def combine_masks(
        self,
        masks: list[np.ndarray[Any, Any]],
        operation: str = "union",
        weights: list[float] | None = None,
    ) -> np.ndarray[Any, Any]:
        """
        Combine multiple masks.

        Args:
            masks: List of masks to combine
            operation: Combination operation ('union', 'intersection', 'weighted')
            weights: Weights for weighted combination

        Returns:
            Combined mask
        """
        if not masks:
            raise ValueError("No masks provided")

        if operation == "union":
            result = np.zeros_like(masks[0])
            for mask in masks:
                result = np.maximum(result, mask)
            return result

        elif operation == "intersection":
            result = np.ones_like(masks[0])
            for mask in masks:
                result = np.minimum(result, mask)
            return result

        elif operation == "weighted":
            if weights is None:
                weights = [1.0 / len(masks)] * len(masks)
            weights = np.array(weights) / np.sum(weights)

            result = np.zeros_like(masks[0])
            for mask, weight in zip(masks, weights, strict=False):
                result += mask * weight
            return result

        else:
            raise ValueError(f"Unknown operation: {operation}")
