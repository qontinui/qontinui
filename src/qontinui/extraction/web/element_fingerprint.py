"""
Element Fingerprint Generator for web extraction.

Generates visual fingerprints from RawElement data for cross-page
element comparison. Uses multiple fingerprint components that can
be weighted/combined for similarity matching.
"""

import hashlib
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .models import (
    BoundingBox,
    ElementFingerprint,
    PositionRegion,
    RawElement,
    SizeClass,
)

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.warning("imagehash not available - visual hashing will use fallback")


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint generation."""

    # Size class thresholds (in pixels)
    size_tiny_threshold: int = 32
    size_small_threshold: int = 100
    size_medium_threshold: int = 300
    size_large_threshold: int = 600

    # Histogram settings
    histogram_bins: int = 8  # Per channel (total = 24 for RGB)

    # Visual hash settings
    hash_size: int = 8  # 8x8 = 64 bit hash

    # Position region grid size
    position_grid_cols: int = 3
    position_grid_rows: int = 3


class ElementFingerprintGenerator:
    """
    Generates visual fingerprints for cross-page element comparison.

    Fingerprint components:
    - Size class: width/height quantized to bins
    - Color histogram: Distribution of colors in element
    - Position region: Where element appears in viewport
    - Content hash: Hash of text + tag + key attributes
    - Visual hash: Perceptual hash of element image

    Each component can be compared independently, allowing
    modular comparison strategies.
    """

    def __init__(self, config: FingerprintConfig | None = None) -> None:
        """Initialize the fingerprint generator."""
        self.config = config or FingerprintConfig()

    def generate_fingerprint(
        self,
        element: RawElement,
        page_screenshot: np.ndarray | None = None,
        viewport_size: tuple[int, int] = (1920, 1080),
    ) -> ElementFingerprint:
        """
        Generate a fingerprint for an element.

        Args:
            element: RawElement to fingerprint.
            page_screenshot: Full page screenshot (BGR numpy array).
                If provided, element image will be cropped for visual analysis.
            viewport_size: (width, height) of viewport for position calculation.

        Returns:
            ElementFingerprint with all computed features.
        """
        # Size classification
        size_class = self._classify_size(element.bbox.width, element.bbox.height)

        # Position region
        position_region = self._classify_position(element.bbox, viewport_size[0], viewport_size[1])

        # Relative position (0.0 to 1.0)
        relative_x = (element.bbox.x + element.bbox.width / 2) / viewport_size[0]
        relative_y = (element.bbox.y + element.bbox.height / 2) / viewport_size[1]
        relative_x = max(0.0, min(1.0, relative_x))
        relative_y = max(0.0, min(1.0, relative_y))

        # Content hash
        content_hash = self._compute_content_hash(element)

        # Initialize fingerprint with basic data
        fingerprint = ElementFingerprint(
            element_id=element.id,
            screenshot_id=element.screenshot_id,
            width=element.bbox.width,
            height=element.bbox.height,
            size_class=size_class,
            position_region=position_region,
            relative_x=relative_x,
            relative_y=relative_y,
            content_hash=content_hash,
            text_length=len(element.text_content or ""),
            has_text=bool(element.text_content),
            has_border=element.border_color is not None and element.border_color[3] > 10,
            has_background=(
                element.background_color is not None and element.background_color[3] > 10
            ),
        )

        # If screenshot provided, compute visual features
        if page_screenshot is not None:
            element_img = self._crop_element(page_screenshot, element.bbox)
            if element_img is not None and element_img.size > 0:
                # Color histogram
                fingerprint.color_histogram = self._compute_color_histogram(element_img)

                # Dominant color
                fingerprint.dominant_color = self._compute_dominant_color(element_img)

                # Visual hash
                fingerprint.visual_hash = self._compute_visual_hash(element_img)

                # Check if has image content
                fingerprint.has_image = self._has_image_content(element_img)

        return fingerprint

    def generate_fingerprints(
        self,
        elements: list[RawElement],
        page_screenshot: np.ndarray | None = None,
        viewport_size: tuple[int, int] = (1920, 1080),
    ) -> list[ElementFingerprint]:
        """
        Generate fingerprints for multiple elements.

        Args:
            elements: List of RawElement objects.
            page_screenshot: Full page screenshot.
            viewport_size: Viewport dimensions.

        Returns:
            List of ElementFingerprint objects.
        """
        fingerprints = []
        for element in elements:
            try:
                fp = self.generate_fingerprint(element, page_screenshot, viewport_size)
                fingerprints.append(fp)
            except Exception as e:
                logger.warning(f"Failed to fingerprint element {element.id}: {e}")

        logger.info(f"Generated {len(fingerprints)} fingerprints")
        return fingerprints

    def _classify_size(self, width: int, height: int) -> SizeClass:
        """Classify element size into bins."""
        # Use the larger dimension for classification
        max_dim = max(width, height)

        if max_dim < self.config.size_tiny_threshold:
            return SizeClass.TINY
        elif max_dim < self.config.size_small_threshold:
            return SizeClass.SMALL
        elif max_dim < self.config.size_medium_threshold:
            return SizeClass.MEDIUM
        elif max_dim < self.config.size_large_threshold:
            return SizeClass.LARGE
        else:
            return SizeClass.XLARGE

    def _classify_position(
        self,
        bbox: BoundingBox,
        viewport_width: int,
        viewport_height: int,
    ) -> PositionRegion:
        """Classify element position into viewport regions."""
        # Calculate center point
        center_x = bbox.x + bbox.width / 2
        center_y = bbox.y + bbox.height / 2

        # Normalize to 0-1 range
        norm_x = center_x / viewport_width if viewport_width > 0 else 0.5
        norm_y = center_y / viewport_height if viewport_height > 0 else 0.5

        # Clamp to valid range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # Map to grid position
        col = int(norm_x * self.config.position_grid_cols)
        row = int(norm_y * self.config.position_grid_rows)

        # Clamp to valid grid indices
        col = min(col, self.config.position_grid_cols - 1)
        row = min(row, self.config.position_grid_rows - 1)

        # Map grid position to PositionRegion enum
        region_map = {
            (0, 0): PositionRegion.TOP_LEFT,
            (1, 0): PositionRegion.TOP_CENTER,
            (2, 0): PositionRegion.TOP_RIGHT,
            (0, 1): PositionRegion.CENTER_LEFT,
            (1, 1): PositionRegion.CENTER,
            (2, 1): PositionRegion.CENTER_RIGHT,
            (0, 2): PositionRegion.BOTTOM_LEFT,
            (1, 2): PositionRegion.BOTTOM_CENTER,
            (2, 2): PositionRegion.BOTTOM_RIGHT,
        }

        return region_map.get((col, row), PositionRegion.CENTER)

    def _compute_content_hash(self, element: RawElement) -> str:
        """
        Compute a hash of element content for text-based matching.

        Includes: tag name, text content, key attributes (id, class, role).
        """
        parts = [
            element.tag_name,
            element.text_content or "",
            element.attributes.get("id", ""),
            element.attributes.get("class", ""),
            element.semantic_role or "",
            element.aria_label or "",
        ]

        content = "|".join(parts)
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]

    def _crop_element(
        self,
        screenshot: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray | None:
        """Crop element region from screenshot."""
        try:
            # Ensure bounds are within image
            h, w = screenshot.shape[:2]
            x1 = max(0, bbox.x)
            y1 = max(0, bbox.y)
            x2 = min(w, bbox.x + bbox.width)
            y2 = min(h, bbox.y + bbox.height)

            if x2 <= x1 or y2 <= y1:
                return None

            return screenshot[y1:y2, x1:x2].copy()

        except Exception as e:
            logger.warning(f"Failed to crop element: {e}")
            return None

    def _compute_color_histogram(self, image: np.ndarray) -> list[int]:
        """
        Compute a color histogram for the element image.

        Returns a 24-bin histogram (8 bins per RGB channel).
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] >= 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale - create pseudo-RGB
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Compute histogram for each channel
            bins = self.config.histogram_bins
            histogram: list[int] = []

            for channel in range(3):
                hist = cv2.calcHist([rgb], [channel], None, [bins], [0, 256])
                # Normalize to percentages
                hist = hist.flatten() / (image.shape[0] * image.shape[1])
                histogram.extend([int(v * 1000) for v in hist])  # Scale for storage

            return histogram

        except Exception as e:
            logger.warning(f"Failed to compute color histogram: {e}")
            return [0] * (self.config.histogram_bins * 3)

    def _compute_dominant_color(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int] | None:
        """
        Compute the dominant color of the element.

        Uses k-means clustering to find the most common color.
        """
        try:
            # Resize for faster processing
            small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)

            # Reshape to list of pixels
            if len(small.shape) == 3:
                pixels = small.reshape(-1, 3).astype(np.float32)
            else:
                pixels = small.reshape(-1, 1).astype(np.float32)

            # K-means to find dominant colors (k=3)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            # Find most common cluster
            label_counts = np.bincount(labels.flatten())
            dominant_idx = np.argmax(label_counts)
            dominant = centers[dominant_idx].astype(int)

            # Convert BGR to RGB
            if len(dominant) == 3:
                return (int(dominant[2]), int(dominant[1]), int(dominant[0]))
            else:
                gray = int(dominant[0])
                return (gray, gray, gray)

        except Exception as e:
            logger.warning(f"Failed to compute dominant color: {e}")
            return None

    def _compute_visual_hash(self, image: np.ndarray) -> str:
        """
        Compute perceptual hash for visual matching.

        Uses pHash algorithm for robust visual fingerprinting.
        """
        try:
            if IMAGEHASH_AVAILABLE:
                # Convert to PIL Image
                if len(image.shape) == 3:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb)
                else:
                    pil_image = Image.fromarray(image)

                # Compute perceptual hash
                phash = imagehash.phash(pil_image, hash_size=self.config.hash_size)
                return str(phash)
            else:
                # Fallback: average hash using OpenCV
                return self._compute_average_hash(image)

        except Exception as e:
            logger.warning(f"Failed to compute visual hash: {e}")
            return ""

    def _compute_average_hash(self, image: np.ndarray) -> str:
        """
        Fallback visual hash using average hash algorithm.

        Doesn't require imagehash library.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Resize to hash_size x hash_size
            size = self.config.hash_size
            resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

            # Compute average
            avg = resized.mean()

            # Create binary hash (1 if pixel > average, else 0)
            binary = (resized > avg).flatten()

            # Convert to hex string
            hash_int = 0
            for bit in binary:
                hash_int = (hash_int << 1) | int(bit)

            return format(hash_int, f"0{size * size // 4}x")

        except Exception as e:
            logger.warning(f"Failed to compute average hash: {e}")
            return ""

    def _has_image_content(self, image: np.ndarray) -> bool:
        """
        Check if element contains significant image content.

        Detects if element has non-uniform colors (not just solid/gradient).
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Check variance - high variance indicates complex content
            variance = gray.var()

            # Check edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            # Element has image content if high variance or many edges
            return variance > 500 or edge_density > 0.1

        except Exception:
            return False


def compute_histogram_similarity(hist1: list[int], hist2: list[int]) -> float:
    """
    Compute similarity between two color histograms.

    Uses Bhattacharyya distance for comparison.

    Args:
        hist1: First histogram (24 bins).
        hist2: Second histogram (24 bins).

    Returns:
        Similarity score 0.0 to 1.0 (1.0 = identical).
    """
    if len(hist1) != len(hist2) or len(hist1) == 0:
        return 0.0

    try:
        # Convert to numpy arrays
        h1 = np.array(hist1, dtype=np.float32)
        h2 = np.array(hist2, dtype=np.float32)

        # Normalize
        h1 = h1 / (h1.sum() + 1e-10)
        h2 = h2 / (h2.sum() + 1e-10)

        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(h1 * h2))

        return float(bc)

    except Exception:
        return 0.0


def compute_hash_similarity(hash1: str, hash2: str) -> float:
    """
    Compute similarity between two visual hashes.

    Uses normalized Hamming distance.

    Args:
        hash1: First hash (hex string).
        hash2: Second hash (hex string).

    Returns:
        Similarity score 0.0 to 1.0 (1.0 = identical).
    """
    if not hash1 or not hash2:
        return 0.0

    if len(hash1) != len(hash2):
        return 0.0

    try:
        # Convert hex to binary
        b1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        b2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        # Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(b1, b2))

        # Normalize to similarity
        return 1.0 - (distance / len(b1))

    except Exception:
        return 0.0


def compute_position_similarity(
    fp1: ElementFingerprint,
    fp2: ElementFingerprint,
) -> float:
    """
    Compute similarity based on relative positions.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.

    Returns:
        Similarity score 0.0 to 1.0 (1.0 = same position).
    """
    # Same region = high similarity
    if fp1.position_region == fp2.position_region:
        region_sim = 1.0
    else:
        region_sim = 0.5

    # Distance between relative positions
    dx = abs(fp1.relative_x - fp2.relative_x)
    dy = abs(fp1.relative_y - fp2.relative_y)
    distance = (dx**2 + dy**2) ** 0.5

    # Max distance is sqrt(2) (corner to corner)
    position_sim = 1.0 - (distance / 1.414)

    return (region_sim + position_sim) / 2


def compute_size_similarity(
    fp1: ElementFingerprint,
    fp2: ElementFingerprint,
) -> float:
    """
    Compute similarity based on element sizes.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.

    Returns:
        Similarity score 0.0 to 1.0 (1.0 = same size).
    """
    # Size class match
    if fp1.size_class != fp2.size_class:
        return 0.0

    # Exact size ratio
    width_ratio = min(fp1.width, fp2.width) / max(fp1.width, fp2.width)
    height_ratio = min(fp1.height, fp2.height) / max(fp1.height, fp2.height)

    return (width_ratio + height_ratio) / 2
