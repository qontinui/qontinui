"""Color Palette Analyzer for GUI Environment Discovery.

Extracts color information from screenshots using:
- K-means clustering for dominant colors
- OCR text proximity analysis for semantic color associations
- Luminance analysis for theme detection
- Before/after comparison for disabled state characteristics
"""

import logging
from collections import Counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.environment import (
    ColorPalette,
    DisabledCharacteristics,
    SemanticColors,
    ThemeType,
)

from qontinui.vision.environment.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)

# Keywords for semantic color association
SEMANTIC_KEYWORDS = {
    "error": ["error", "failed", "failure", "invalid", "wrong", "danger", "critical"],
    "success": ["success", "done", "complete", "passed", "valid", "ok", "saved"],
    "warning": ["warning", "caution", "attention", "alert", "notice"],
    "info": ["info", "information", "note", "tip", "hint"],
}


class ColorPaletteAnalyzer(BaseAnalyzer[ColorPalette]):
    """Analyzes screenshots to extract color palette information.

    Uses K-means clustering to identify dominant colors and associates
    colors with semantic meanings based on proximity to text keywords.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        sample_ratio: float = 0.1,
        edge_aware: bool = True,
    ) -> None:
        """Initialize the color palette analyzer.

        Args:
            n_clusters: Number of clusters for K-means (default 10).
            sample_ratio: Fraction of pixels to sample (default 0.1).
            edge_aware: Whether to weight edge pixels more (default True).
        """
        super().__init__("ColorPaletteAnalyzer")
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.edge_aware = edge_aware

    async def analyze(
        self,
        screenshots: list[NDArray[np.uint8]],
        ocr_results: list[list[dict[str, Any]]] | None = None,
        disabled_pairs: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]] | None = None,
        **kwargs: Any,
    ) -> ColorPalette:
        """Analyze screenshots to extract color palette.

        Args:
            screenshots: List of screenshots as numpy arrays (BGR format).
            ocr_results: Optional OCR results for semantic association.
                Each result is a list of dicts with 'text' and 'bbox' keys.
            disabled_pairs: Optional list of (enabled, disabled) screenshot pairs
                for learning disabled characteristics.

        Returns:
            ColorPalette with extracted color information.
        """
        self.reset()

        if not screenshots:
            return ColorPalette(confidence=0.0)

        self._log_progress(f"Analyzing {len(screenshots)} screenshots")

        # Extract dominant colors using K-means
        all_pixels = self._collect_pixels(screenshots)
        dominant_colors = self._extract_dominant_colors(all_pixels)

        # Detect theme type
        theme_type = self._detect_theme_type(dominant_colors, screenshots)

        # Extract semantic colors
        semantic_colors = await self._extract_semantic_colors(
            screenshots, dominant_colors, ocr_results
        )

        # Learn disabled characteristics if pairs provided
        disabled_chars = None
        if disabled_pairs:
            disabled_chars = self._learn_disabled_characteristics(disabled_pairs)

        self._screenshots_analyzed = len(screenshots)
        self.confidence = self._calculate_confidence(
            len(screenshots),
            min_samples=2,
            optimal_samples=10,
        )

        return ColorPalette(
            dominant_colors=dominant_colors,
            semantic_colors=semantic_colors,
            theme_type=theme_type,
            disabled_characteristics=disabled_chars,
            screenshots_analyzed=len(screenshots),
            confidence=self.confidence,
        )

    def _collect_pixels(
        self,
        screenshots: list[NDArray[np.uint8]],
    ) -> NDArray[np.uint8]:
        """Collect pixel samples from all screenshots.

        Args:
            screenshots: List of BGR screenshots.

        Returns:
            Array of sampled pixels in RGB format.
        """
        all_pixels = []

        for screenshot in screenshots:
            screenshot = self._ensure_bgr(screenshot)
            h, w = screenshot.shape[:2]
            total_pixels = h * w

            # Calculate number of samples
            n_samples = int(total_pixels * self.sample_ratio)
            n_samples = max(1000, min(n_samples, 50000))  # Clamp to reasonable range

            if self.edge_aware:
                # Get edge mask using Sobel
                edge_pixels, non_edge_pixels = self._get_edge_weighted_samples(
                    screenshot, n_samples
                )
                pixels = np.vstack([edge_pixels, non_edge_pixels])
            else:
                # Random sampling
                flat_pixels = screenshot.reshape(-1, 3)
                indices = np.random.choice(len(flat_pixels), n_samples, replace=False)
                pixels = flat_pixels[indices]

            # Convert BGR to RGB
            pixels = pixels[:, ::-1]
            all_pixels.append(pixels)

        return np.vstack(all_pixels)

    def _get_edge_weighted_samples(
        self,
        screenshot: NDArray[np.uint8],
        n_samples: int,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Sample pixels with higher weight on edges.

        Args:
            screenshot: BGR screenshot.
            n_samples: Total number of samples.

        Returns:
            Tuple of (edge_pixels, non_edge_pixels).
        """
        try:
            import cv2

            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # Threshold for edges
            edge_threshold = np.percentile(edge_magnitude, 80)
            edge_mask = edge_magnitude > edge_threshold

            # Sample more from edges (60% from edges, 40% from non-edges)
            n_edge_samples = int(n_samples * 0.6)
            n_non_edge_samples = n_samples - n_edge_samples

            edge_indices = np.where(edge_mask.flatten())[0]
            non_edge_indices = np.where(~edge_mask.flatten())[0]

            flat_pixels = screenshot.reshape(-1, 3)

            # Sample from edges
            if len(edge_indices) > 0:
                edge_sample_idx = np.random.choice(
                    edge_indices,
                    min(n_edge_samples, len(edge_indices)),
                    replace=False,
                )
                edge_pixels = flat_pixels[edge_sample_idx]
            else:
                edge_pixels = np.array([]).reshape(0, 3)

            # Sample from non-edges
            if len(non_edge_indices) > 0:
                non_edge_sample_idx = np.random.choice(
                    non_edge_indices,
                    min(n_non_edge_samples, len(non_edge_indices)),
                    replace=False,
                )
                non_edge_pixels = flat_pixels[non_edge_sample_idx]
            else:
                non_edge_pixels = np.array([]).reshape(0, 3)

            return edge_pixels, non_edge_pixels

        except ImportError:
            # Fallback to random sampling if cv2 not available
            flat_pixels = screenshot.reshape(-1, 3)
            indices = np.random.choice(len(flat_pixels), n_samples, replace=False)
            return flat_pixels[indices], np.array([]).reshape(0, 3)

    def _extract_dominant_colors(
        self,
        pixels: NDArray[np.uint8],
    ) -> list[str]:
        """Extract dominant colors using K-means clustering.

        Args:
            pixels: Array of RGB pixels.

        Returns:
            List of hex color strings sorted by frequency.
        """
        try:
            from sklearn.cluster import KMeans

            # Fit K-means
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
            kmeans.fit(pixels)

            # Get cluster centers and counts
            centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            counts = Counter(labels)

            # Sort by frequency
            sorted_indices = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)

            # Convert to hex colors
            colors = []
            for idx in sorted_indices:
                r, g, b = centers[idx]
                colors.append(self._rgb_to_hex(int(r), int(g), int(b)))

            return colors

        except ImportError:
            logger.warning("sklearn not available, using simple color quantization")
            return self._simple_color_quantization(pixels)

    def _simple_color_quantization(
        self,
        pixels: NDArray[np.uint8],
    ) -> list[str]:
        """Simple color quantization without sklearn.

        Args:
            pixels: Array of RGB pixels.

        Returns:
            List of hex color strings.
        """
        # Quantize to 32 levels per channel
        quantized = (pixels // 8) * 8
        # Convert to tuples and count
        pixel_tuples = [tuple(p) for p in quantized]
        counts = Counter(pixel_tuples)

        # Get top colors
        top_colors = counts.most_common(self.n_clusters)
        return [self._rgb_to_hex(int(r), int(g), int(b)) for (r, g, b), _ in top_colors]

    def _detect_theme_type(
        self,
        dominant_colors: list[str],
        screenshots: list[NDArray[np.uint8]],
    ) -> ThemeType:
        """Detect whether the app uses a dark or light theme.

        Args:
            dominant_colors: List of dominant hex colors.
            screenshots: Original screenshots for background detection.

        Returns:
            ThemeType enum value.
        """
        # Calculate average luminance of dominant colors
        luminances = []
        for hex_color in dominant_colors[:5]:  # Top 5 colors
            r, g, b = self._hex_to_rgb(hex_color)
            luminances.append(self._calculate_luminance(r, g, b))

        avg_luminance = sum(luminances) / len(luminances) if luminances else 0.5

        # Also check the most common color in large regions (likely background)
        background_luminance = self._estimate_background_luminance(screenshots)

        # Weight background luminance more heavily
        weighted_luminance = 0.3 * avg_luminance + 0.7 * background_luminance

        if weighted_luminance < 0.3:
            return ThemeType.DARK
        elif weighted_luminance > 0.7:
            return ThemeType.LIGHT
        else:
            return ThemeType.MIXED

    def _estimate_background_luminance(
        self,
        screenshots: list[NDArray[np.uint8]],
    ) -> float:
        """Estimate background luminance from screenshots.

        Args:
            screenshots: List of BGR screenshots.

        Returns:
            Average background luminance (0.0-1.0).
        """
        luminances = []

        for screenshot in screenshots[:3]:  # Sample first 3 screenshots
            screenshot = self._ensure_bgr(screenshot)
            h, w = screenshot.shape[:2]

            # Sample corners and edges (likely background)
            regions = [
                screenshot[0:50, 0:50],  # Top-left
                screenshot[0:50, w - 50 : w],  # Top-right
                screenshot[h - 50 : h, 0:50],  # Bottom-left
                screenshot[h - 50 : h, w - 50 : w],  # Bottom-right
            ]

            for region in regions:
                # Get most common color in region
                flat = region.reshape(-1, 3)
                # Simple mode approximation
                avg_color = flat.mean(axis=0).astype(int)
                b, g, r = avg_color
                luminances.append(self._calculate_luminance(r, g, b))

        return sum(luminances) / len(luminances) if luminances else 0.5

    async def _extract_semantic_colors(
        self,
        screenshots: list[NDArray[np.uint8]],
        dominant_colors: list[str],
        ocr_results: list[list[dict[str, Any]]] | None,
    ) -> SemanticColors:
        """Extract semantic color associations.

        Args:
            screenshots: List of BGR screenshots.
            dominant_colors: Dominant colors from K-means.
            ocr_results: Optional OCR results for text-based association.

        Returns:
            SemanticColors with associations.
        """
        semantic = SemanticColors()

        # Background is typically the most dominant color
        if dominant_colors:
            semantic.background = dominant_colors[0]

        # Try to identify text color (high contrast with background)
        if dominant_colors and len(dominant_colors) > 1:
            bg_r, bg_g, bg_b = self._hex_to_rgb(dominant_colors[0])
            bg_luminance = self._calculate_luminance(bg_r, bg_g, bg_b)

            for hex_color in dominant_colors[1:]:
                r, g, b = self._hex_to_rgb(hex_color)
                luminance = self._calculate_luminance(r, g, b)

                # High contrast = likely text
                if abs(luminance - bg_luminance) > 0.4:
                    if semantic.text_primary is None:
                        semantic.text_primary = hex_color
                    elif semantic.text_secondary is None:
                        semantic.text_secondary = hex_color
                        break

        # Associate colors with semantic keywords if OCR available
        if ocr_results:
            keyword_colors = self._associate_colors_with_keywords(screenshots, ocr_results)
            if "error" in keyword_colors:
                semantic.error = keyword_colors["error"]
            if "success" in keyword_colors:
                semantic.success = keyword_colors["success"]
            if "warning" in keyword_colors:
                semantic.warning = keyword_colors["warning"]
            if "info" in keyword_colors:
                semantic.info = keyword_colors["info"]

        # Identify accent color (saturated, not background/text)
        semantic.accent = self._find_accent_color(dominant_colors, semantic)

        return semantic

    def _associate_colors_with_keywords(
        self,
        screenshots: list[NDArray[np.uint8]],
        ocr_results: list[list[dict[str, Any]]],
    ) -> dict[str, str]:
        """Associate colors with semantic keywords based on text proximity.

        Args:
            screenshots: BGR screenshots.
            ocr_results: OCR results with text and bounding boxes.

        Returns:
            Dictionary mapping semantic type to hex color.
        """
        associations: dict[str, list[str]] = {k: [] for k in SEMANTIC_KEYWORDS}

        for screenshot, ocr_result in zip(screenshots, ocr_results):
            screenshot = self._ensure_bgr(screenshot)

            for item in ocr_result:
                text = item.get("text", "").lower()
                bbox = item.get("bbox")

                if not bbox:
                    continue

                # Check which semantic category this text matches
                for category, keywords in SEMANTIC_KEYWORDS.items():
                    if any(kw in text for kw in keywords):
                        # Sample color near this text
                        color = self._sample_color_near_bbox(screenshot, bbox)
                        if color:
                            associations[category].append(color)

        # Get most common color for each category
        result = {}
        for category, colors in associations.items():
            if colors:
                # Get mode
                counts = Counter(colors)
                most_common = counts.most_common(1)[0][0]
                result[category] = most_common

        return result

    def _sample_color_near_bbox(
        self,
        screenshot: NDArray[np.uint8],
        bbox: tuple[int, int, int, int] | list[int],
    ) -> str | None:
        """Sample the dominant color near a bounding box.

        Args:
            screenshot: BGR screenshot.
            bbox: Bounding box (x, y, width, height) or (x1, y1, x2, y2).

        Returns:
            Hex color string or None.
        """
        try:
            # Handle different bbox formats
            if len(bbox) == 4:
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # (x1, y1, x2, y2) format
                    x1, y1, x2, y2 = bbox
                else:
                    # (x, y, width, height) format
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                return None

            # Expand region slightly
            h_img, w_img = screenshot.shape[:2]
            padding = 10
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(w_img, int(x2) + padding)
            y2 = min(h_img, int(y2) + padding)

            if x2 <= x1 or y2 <= y1:
                return None

            region = screenshot[y1:y2, x1:x2]
            if region.size == 0:
                return None

            # Get average color (simple approach)
            avg_color = region.reshape(-1, 3).mean(axis=0).astype(int)
            b, g, r = avg_color
            return self._rgb_to_hex(int(r), int(g), int(b))

        except Exception as e:
            logger.debug(f"Error sampling color near bbox: {e}")
            return None

    def _find_accent_color(
        self,
        dominant_colors: list[str],
        semantic: SemanticColors,
    ) -> str | None:
        """Find an accent color (saturated, distinct from bg/text).

        Args:
            dominant_colors: List of dominant hex colors.
            semantic: Already identified semantic colors.

        Returns:
            Hex color string or None.
        """
        excluded = {
            semantic.background,
            semantic.text_primary,
            semantic.text_secondary,
        }

        for hex_color in dominant_colors:
            if hex_color in excluded:
                continue

            r, g, b = self._hex_to_rgb(hex_color)

            # Check saturation (simplified HSV saturation)
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            if max_c == 0:
                continue

            saturation = (max_c - min_c) / max_c

            # Accent colors are typically saturated
            if saturation > 0.4:
                return hex_color

        return None

    def _learn_disabled_characteristics(
        self,
        disabled_pairs: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    ) -> DisabledCharacteristics:
        """Learn disabled state visual characteristics from before/after pairs.

        Args:
            disabled_pairs: List of (enabled_screenshot, disabled_screenshot) pairs.

        Returns:
            DisabledCharacteristics with measured values.
        """
        saturation_reductions = []
        brightness_changes = []

        for enabled, disabled in disabled_pairs:
            enabled = self._ensure_bgr(enabled)
            disabled = self._ensure_bgr(disabled)

            # Compare average saturation
            enabled_sat = self._calculate_avg_saturation(enabled)
            disabled_sat = self._calculate_avg_saturation(disabled)

            if enabled_sat > 0:
                sat_reduction = disabled_sat / enabled_sat
                saturation_reductions.append(sat_reduction)

            # Compare brightness
            enabled_bright = self._calculate_avg_brightness(enabled)
            disabled_bright = self._calculate_avg_brightness(disabled)

            if enabled_bright > 0:
                bright_change = (disabled_bright - enabled_bright) / enabled_bright
                brightness_changes.append(bright_change)

        # Calculate averages
        avg_sat_reduction = (
            sum(saturation_reductions) / len(saturation_reductions)
            if saturation_reductions
            else 0.6
        )
        avg_bright_change = (
            sum(brightness_changes) / len(brightness_changes) if brightness_changes else 0.0
        )

        return DisabledCharacteristics(
            saturation_reduction=max(0.0, min(1.0, avg_sat_reduction)),
            opacity_reduction=max(0.0, min(1.0, 1.0 - avg_sat_reduction)),
            brightness_change=avg_bright_change,
            observed_samples=len(disabled_pairs),
        )

    def _calculate_avg_saturation(self, image: NDArray[np.uint8]) -> float:
        """Calculate average saturation of an image.

        Args:
            image: BGR image.

        Returns:
            Average saturation (0.0-1.0).
        """
        try:
            import cv2

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return float(hsv[:, :, 1].mean() / 255.0)
        except ImportError:
            # Simplified saturation calculation
            image = image.astype(float)
            max_c = image.max(axis=2)
            min_c = image.min(axis=2)
            mask = max_c > 0
            saturation = np.zeros_like(max_c)
            saturation[mask] = (max_c[mask] - min_c[mask]) / max_c[mask]
            return float(saturation.mean())

    def _calculate_avg_brightness(self, image: NDArray[np.uint8]) -> float:
        """Calculate average brightness of an image.

        Args:
            image: BGR image.

        Returns:
            Average brightness (0.0-1.0).
        """
        return float(image.mean() / 255.0)
