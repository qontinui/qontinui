"""
Background Removal for State Discovery

Removes dynamic backgrounds from screenshots to enable robust pixel-based
comparison of UI elements that remain fixed in position.
"""

import base64
import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BackgroundRemovalConfig:
    """Configuration for background removal"""

    # Detection strategies (enable/disable each)
    use_temporal_variance: bool = True  # Pixels that change between screenshots
    use_edge_density: bool = True  # Low edge density regions
    use_uniformity: bool = True  # Large uniform color regions

    # Temporal variance thresholds
    variance_threshold: float = 20.0  # Std dev of pixel values across screenshots
    min_screenshots_for_variance: int = 3  # Minimum screenshots needed for variance

    # Edge density thresholds
    edge_density_threshold: float = 0.05  # Edges per pixel (lower = more background)
    edge_kernel_size: int = 3  # Kernel size for edge detection

    # Uniformity thresholds
    uniformity_threshold: float = 15.0  # Max std dev within region
    uniformity_region_size: int = 20  # Size of region for uniformity check

    # Morphological operations (cleanup)
    apply_morphology: bool = True
    morphology_kernel_size: int = 3  # For opening/closing operations
    min_foreground_region_size: int = 50  # Minimum connected component size

    # Output format
    foreground_alpha: int = 255  # Fully opaque for foreground
    background_alpha: int = 0  # Fully transparent for background


class BackgroundRemovalAnalyzer:
    """Remove dynamic backgrounds from screenshots for State Discovery"""

    def __init__(self, config: BackgroundRemovalConfig | None = None):
        """
        Initialize BackgroundRemovalAnalyzer

        Args:
            config: Configuration for background removal. Uses defaults if None.
        """
        self.config = config or BackgroundRemovalConfig()
        logger.info("BackgroundRemovalAnalyzer initialized")

    def remove_backgrounds(
        self, screenshots: list[np.ndarray], debug: bool = False
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        """
        Remove backgrounds from screenshots, returning RGBA versions

        Args:
            screenshots: List of BGR screenshot images (numpy arrays)
            debug: If True, return detailed debug information

        Returns:
            Tuple of:
            - List of RGBA images with background pixels made transparent
            - Dictionary with statistics and debug information
        """
        if not screenshots:
            logger.warning("No screenshots provided")
            return [], {}

        logger.info(f"Starting background removal for {len(screenshots)} screenshots")

        # Build composite background mask
        background_mask = self._identify_background(screenshots)

        # Apply mask to each screenshot
        masked_screenshots = []
        for _idx, screenshot in enumerate(screenshots):
            masked = self._apply_background_mask(screenshot, background_mask)
            masked_screenshots.append(masked)

        # Calculate statistics
        stats = self._calculate_statistics(background_mask, screenshots)

        if debug:
            stats["background_mask"] = background_mask
            stats["foreground_pixel_count"] = np.sum(background_mask == 0)
            stats["background_pixel_count"] = np.sum(background_mask == 255)

        logger.info(
            f"Background removal complete. Foreground: {stats.get('foreground_percentage', 0):.1f}%"
        )

        return masked_screenshots, stats

    def _identify_background(self, screenshots: list[np.ndarray]) -> np.ndarray:
        """
        Identify pixels that are likely background

        Args:
            screenshots: List of screenshot images

        Returns:
            Binary mask where 255 = background, 0 = foreground
        """
        height, width = screenshots[0].shape[:2]
        background_mask = np.zeros((height, width), dtype=np.uint8)

        # Strategy 1: Temporal variance (pixels that change between screenshots)
        if (
            self.config.use_temporal_variance
            and len(screenshots) >= self.config.min_screenshots_for_variance
        ):
            variance_mask = self._detect_by_temporal_variance(screenshots)
            background_mask = cv2.bitwise_or(background_mask, variance_mask).astype(np.uint8)
            logger.debug(
                f"Temporal variance detected {np.sum(variance_mask > 0)} background pixels"
            )

        # Strategy 2: Edge density (low edge regions are likely background)
        if self.config.use_edge_density:
            edge_mask = self._detect_by_edge_density(screenshots[0])
            background_mask = cv2.bitwise_or(background_mask, edge_mask).astype(np.uint8)
            logger.debug(
                f"Edge density detected {np.sum(edge_mask > 0)} background pixels"
            )

        # Strategy 3: Uniformity (large uniform regions are likely background)
        if self.config.use_uniformity:
            uniformity_mask = self._detect_by_uniformity(screenshots[0])
            background_mask = cv2.bitwise_or(background_mask, uniformity_mask).astype(np.uint8)
            logger.debug(
                f"Uniformity detected {np.sum(uniformity_mask > 0)} background pixels"
            )

        # Apply morphological operations to clean up mask
        if self.config.apply_morphology:
            background_mask = self._apply_morphological_cleanup(background_mask)

        return background_mask

    def _detect_by_temporal_variance(self, screenshots: list[np.ndarray]) -> np.ndarray:
        """
        Detect background by finding pixels that vary across screenshots

        Pixels that change significantly are likely background (e.g., animated
        backgrounds, gradients that shift, etc.)

        Args:
            screenshots: List of screenshot images

        Returns:
            Binary mask where 255 = background (high variance)
        """
        height, width = screenshots[0].shape[:2]

        # Convert all screenshots to grayscale for variance calculation
        gray_screenshots = []
        for screenshot in screenshots:
            if len(screenshot.shape) == 3:
                gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            else:
                gray = screenshot
            gray_screenshots.append(gray.astype(np.float32))

        # Stack into 3D array (height, width, num_screenshots)
        stack = np.stack(gray_screenshots, axis=2)

        # Calculate standard deviation across screenshots for each pixel
        pixel_std = np.std(stack, axis=2)

        # Pixels with high variance are background
        variance_mask: np.ndarray = (pixel_std > self.config.variance_threshold).astype(
            np.uint8
        ) * 255

        return variance_mask

    def _detect_by_edge_density(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Detect background by finding regions with low edge density

        Background regions typically have fewer edges than UI elements

        Args:
            screenshot: Single screenshot image

        Returns:
            Binary mask where 255 = background (low edge density)
        """
        height, width = screenshot.shape[:2]

        # Convert to grayscale
        if len(screenshot.shape) == 3:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density in regions
        region_size = self.config.uniformity_region_size
        edge_density = np.zeros((height, width), dtype=np.float32)

        for y in range(0, height, region_size // 2):
            for x in range(0, width, region_size // 2):
                y_end = min(y + region_size, height)
                x_end = min(x + region_size, width)

                region_edges = edges[y:y_end, x:x_end]
                density = np.sum(region_edges > 0) / (region_edges.size)

                edge_density[y:y_end, x:x_end] = density

        # Low edge density = background
        edge_mask = (edge_density < self.config.edge_density_threshold).astype(
            np.uint8
        ) * 255

        return edge_mask

    def _detect_by_uniformity(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Detect background by finding large uniform color regions

        Args:
            screenshot: Single screenshot image

        Returns:
            Binary mask where 255 = background (uniform regions)
        """
        height, width = screenshot.shape[:2]

        # Convert to LAB color space for better color uniformity detection
        if len(screenshot.shape) == 3:
            lab = cv2.cvtColor(screenshot, cv2.COLOR_BGR2LAB)
        else:
            lab = cv2.cvtColor(
                cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB
            )

        # Calculate local standard deviation
        region_size = self.config.uniformity_region_size
        uniformity_mask = np.zeros((height, width), dtype=np.uint8)

        for y in range(0, height, region_size // 2):
            for x in range(0, width, region_size // 2):
                y_end = min(y + region_size, height)
                x_end = min(x + region_size, width)

                region = lab[y:y_end, x:x_end]

                # Calculate std dev for each channel
                std_devs = [np.std(region[:, :, i]) for i in range(3)]
                avg_std = np.mean(std_devs)

                # Mark as background if very uniform
                if avg_std < self.config.uniformity_threshold:
                    uniformity_mask[y:y_end, x:x_end] = 255

        return uniformity_mask

    def _apply_morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the mask

        This removes small noise and smooths boundaries

        Args:
            mask: Binary mask

        Returns:
            Cleaned binary mask
        """
        kernel = np.ones(
            (self.config.morphology_kernel_size, self.config.morphology_kernel_size),
            np.uint8,
        )

        # Opening: removes small foreground noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Closing: fills small holes in foreground
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Invert to work with foreground
        foreground_mask = cv2.bitwise_not(mask)

        # Find connected components in foreground
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask, connectivity=8
        )

        # Create cleaned foreground mask
        cleaned_foreground = np.zeros_like(foreground_mask)

        # Keep only components larger than minimum size
        for label in range(1, num_labels):  # Skip background (0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.config.min_foreground_region_size:
                cleaned_foreground[labels == label] = 255

        # Invert back to background mask
        cleaned_mask = cv2.bitwise_not(cleaned_foreground)

        return cleaned_mask

    def _apply_background_mask(
        self, screenshot: np.ndarray, background_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply transparency mask to remove background

        Args:
            screenshot: Original BGR screenshot
            background_mask: Binary mask where 255 = background

        Returns:
            RGBA image with background made transparent
        """
        height, width = screenshot.shape[:2]

        # Convert BGR to BGRA
        if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
            rgba = cv2.cvtColor(screenshot, cv2.COLOR_BGR2BGRA)
        elif len(screenshot.shape) == 2:
            rgba = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGRA)
        else:
            rgba = screenshot.copy()

        # Create alpha channel: 0 where background, 255 where foreground
        alpha_channel = np.where(
            background_mask == 255,
            self.config.background_alpha,
            self.config.foreground_alpha,
        ).astype(np.uint8)

        # Apply alpha channel
        rgba[:, :, 3] = alpha_channel

        return rgba

    def _calculate_statistics(
        self, background_mask: np.ndarray, screenshots: list[np.ndarray]
    ) -> dict[str, Any]:
        """
        Calculate statistics about background removal

        Args:
            background_mask: Binary mask
            screenshots: Original screenshots

        Returns:
            Dictionary with statistics
        """
        total_pixels = background_mask.size
        background_pixels = np.sum(background_mask == 255)
        foreground_pixels = total_pixels - background_pixels

        return {
            "total_pixels": int(total_pixels),
            "background_pixels": int(background_pixels),
            "foreground_pixels": int(foreground_pixels),
            "background_percentage": float(background_pixels / total_pixels * 100),
            "foreground_percentage": float(foreground_pixels / total_pixels * 100),
            "num_screenshots": len(screenshots),
            "image_size": (screenshots[0].shape[1], screenshots[0].shape[0]),
        }

    def visualize_mask(
        self, screenshot: np.ndarray, background_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create visualization showing background mask overlay

        Args:
            screenshot: Original screenshot
            background_mask: Background mask

        Returns:
            Visualization image with red overlay on background regions
        """
        if len(screenshot.shape) == 2:
            vis = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
        else:
            vis = screenshot.copy()

        # Create red overlay for background
        overlay = np.zeros_like(vis)
        overlay[:, :] = (0, 0, 255)  # Red in BGR

        # Apply overlay only where background
        mask_3channel = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        vis = np.where(
            mask_3channel == 255, cv2.addWeighted(vis, 0.6, overlay, 0.4, 0), vis
        )

        return vis


def create_default_config() -> BackgroundRemovalConfig:
    """
    Create default configuration for background removal

    Returns:
        Default BackgroundRemovalConfig
    """
    return BackgroundRemovalConfig()


def remove_backgrounds_simple(
    screenshots: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Simple convenience function to remove backgrounds with default settings

    Args:
        screenshots: List of BGR screenshots

    Returns:
        List of RGBA screenshots with backgrounds removed
    """
    analyzer = BackgroundRemovalAnalyzer()
    masked_screenshots, _ = analyzer.remove_backgrounds(screenshots)
    return masked_screenshots


# Base64 encoding/decoding utilities for web API integration


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 string to an OpenCV image (numpy array).

    Args:
        base64_string: Base64 encoded image string (with or without data URL prefix)

    Returns:
        BGR image as numpy array

    Raises:
        ValueError: If the image cannot be decoded
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)

        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image from base64 string")

        return img

    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid base64 image data: {e}") from e


def encode_image_to_base64(image: np.ndarray, format: str = "png") -> str:
    """
    Encode an OpenCV image (numpy array) to base64 string.

    Args:
        image: Image as numpy array (BGR or BGRA)
        format: Image format ('png', 'jpg', etc.). Default is 'png' which supports transparency.

    Returns:
        Base64 encoded image string with data URL prefix

    Raises:
        ValueError: If the image cannot be encoded
    """
    try:
        # Encode image to specified format
        success, buffer = cv2.imencode(f".{format}", image)

        if not success:
            raise ValueError(f"Failed to encode image to {format} format")

        # Convert to base64
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Add data URL prefix
        mime_type = "image/png" if format == "png" else f"image/{format}"
        return f"data:{mime_type};base64,{img_b64}"

    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise ValueError(f"Image encoding failed: {e}") from e


def remove_backgrounds_from_base64(
    base64_screenshots: list[str],
    config: BackgroundRemovalConfig | None = None,
    debug: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """
    Remove backgrounds from base64-encoded screenshots.

    This is a convenience function for web APIs that work with base64 strings.

    Args:
        base64_screenshots: List of base64 encoded screenshot strings
        config: Configuration for background removal. Uses defaults if None.
        debug: If True, include debug information (note: debug mask won't be in base64)

    Returns:
        Tuple of:
        - List of base64 encoded RGBA images with backgrounds removed
        - Dictionary with statistics and debug information

    Raises:
        ValueError: If any screenshot cannot be decoded or processed
    """
    try:
        # Decode all base64 screenshots to numpy arrays
        screenshots = []
        for idx, b64_str in enumerate(base64_screenshots):
            try:
                img = decode_base64_image(b64_str)
                screenshots.append(img)
            except ValueError as e:
                raise ValueError(f"Failed to decode screenshot {idx}: {e}") from e

        # Process screenshots
        analyzer = BackgroundRemovalAnalyzer(config)
        masked_screenshots, stats = analyzer.remove_backgrounds(
            screenshots, debug=debug
        )

        # Encode results back to base64
        base64_results = []
        for masked_img in masked_screenshots:
            b64_str = encode_image_to_base64(masked_img, format="png")
            base64_results.append(b64_str)

        # Encode debug mask if present
        if debug and "background_mask" in stats:
            mask = stats["background_mask"]
            stats["background_mask_base64"] = encode_image_to_base64(mask, format="png")
            # Remove the numpy array version to keep the response serializable
            del stats["background_mask"]

        return base64_results, stats

    except Exception as e:
        logger.error(f"Background removal from base64 failed: {e}", exc_info=True)
        raise ValueError(f"Background removal processing failed: {e}") from e
