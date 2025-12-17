"""Element detector base implementation.

This module provides the core element detection interface and base implementations
for discovering GUI elements in screenshots. Detectors identify potential UI elements
that can become StateImages in state structure construction.

Detection strategies:
- Feature-based detection: Uses ORB/SIFT to find visually distinct regions
- OCR-based detection: Finds text elements and classifies them
- ML-based detection: Uses trained models for element classification
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class DetectedElement:
    """Represents a detected UI element.

    Attributes:
        element_type: Type of element (button, text_field, icon, etc.)
        bounds: Bounding box (x, y, width, height)
        confidence: Detection confidence score (0.0 to 1.0)
        features: Additional features or metadata
        image: Optional cropped image of the element
    """

    element_type: str
    bounds: tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    features: dict | None = None
    image: np.ndarray | None = None

    def __repr__(self) -> str:
        """String representation of detected element."""
        return (
            f"DetectedElement(type={self.element_type}, "
            f"bounds={self.bounds}, confidence={self.confidence:.3f})"
        )


class ElementDetector(ABC):
    """Abstract base class for element detection implementations.

    Subclasses should implement specific detection strategies such as:
    - Template matching
    - Feature-based detection
    - OCR-based text detection
    - Machine learning classification
    """

    @abstractmethod
    def detect(self, screenshot: np.ndarray) -> list[DetectedElement]:
        """Detect elements in a screenshot.

        Args:
            screenshot: Screenshot image as numpy array

        Returns:
            List of detected elements
        """
        pass

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure detector parameters.

        Args:
            **kwargs: Configuration parameters specific to the detector
        """
        pass


class CompositeElementDetector(ElementDetector):
    """Combines multiple detection methods.

    This detector runs multiple detection strategies and combines their results,
    handling overlapping detections and selecting the best results.
    """

    def __init__(self, detectors: list[ElementDetector] | None = None):
        """Initialize composite detector.

        Args:
            detectors: List of element detectors to combine
        """
        self.detectors = detectors or []
        self.confidence_threshold = 0.7
        self.iou_threshold = 0.5  # For non-maximum suppression

    def detect(self, screenshot: np.ndarray) -> list[DetectedElement]:
        """Detect elements using all configured detectors.

        Args:
            screenshot: Screenshot image

        Returns:
            Combined list of detected elements with duplicates removed
        """
        all_elements: list[DetectedElement] = []

        # Run all detectors
        for detector in self.detectors:
            elements = detector.detect(screenshot)
            all_elements.extend(elements)

        # Filter by confidence
        filtered = [e for e in all_elements if e.confidence >= self.confidence_threshold]

        # Apply non-maximum suppression to remove overlaps
        final_elements = self._non_maximum_suppression(filtered)

        return final_elements

    def configure(self, **kwargs) -> None:
        """Configure composite detector.

        Args:
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        if "confidence_threshold" in kwargs:
            self.confidence_threshold = kwargs["confidence_threshold"]
        if "iou_threshold" in kwargs:
            self.iou_threshold = kwargs["iou_threshold"]

    def add_detector(self, detector: ElementDetector) -> None:
        """Add a detector to the composite.

        Args:
            detector: Element detector to add
        """
        self.detectors.append(detector)

    def _non_maximum_suppression(self, elements: list[DetectedElement]) -> list[DetectedElement]:
        """Apply non-maximum suppression to remove overlapping detections.

        Args:
            elements: List of detected elements

        Returns:
            Filtered list with overlaps removed
        """
        if not elements:
            return []

        # Sort by confidence (descending)
        sorted_elements = sorted(elements, key=lambda e: e.confidence, reverse=True)

        keep = []
        while sorted_elements:
            # Keep the highest confidence element
            current = sorted_elements.pop(0)
            keep.append(current)

            # Remove overlapping elements
            sorted_elements = [
                e
                for e in sorted_elements
                if self._calculate_iou(current.bounds, e.bounds) < self.iou_threshold
            ]

        return keep

    def _calculate_iou(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes.

        Args:
            box1: First bounding box (x, y, width, height)
            box2: Second bounding box (x, y, width, height)

        Returns:
            IoU score between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


class FeatureDetector(ElementDetector):
    """Feature-based GUI element detector using ORB/SIFT.

    Discovers potential UI elements in screenshots by detecting visually distinct
    regions with strong feature points. These regions are candidates for becoming
    StateImages in the state structure.

    The detector identifies:
    - Icons and buttons (clusters of features in small regions)
    - Interactive elements (high feature density areas)
    - Visual landmarks (distinctive regions for state identification)

    Uses ORB as primary algorithm (fast, patent-free) with optional SIFT fallback
    for difficult cases.
    """

    def __init__(self) -> None:
        """Initialize feature detector with default parameters."""
        self.min_features = 10  # Minimum features to consider a region
        self.max_features = 500  # Maximum features per image
        self.min_region_size = (20, 20)  # Minimum region size
        self.max_region_size = (500, 500)  # Maximum region size
        self.cluster_distance = 50  # Pixels for feature clustering
        self.confidence_threshold = 0.6
        self._orb: cv2.ORB | None = None

    @property
    def orb(self) -> cv2.ORB:
        """Lazy-load ORB detector."""
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=self.max_features)  # type: ignore[attr-defined]
        return self._orb

    def detect(self, screenshot: np.ndarray) -> list[DetectedElement]:
        """Detect GUI elements using feature detection.

        Finds visually distinct regions that could be UI elements by:
        1. Detecting ORB features across the image
        2. Clustering nearby features into regions
        3. Filtering regions by size and feature density
        4. Scoring regions by visual distinctiveness

        Args:
            screenshot: Screenshot image as numpy array (BGR or grayscale)

        Returns:
            List of detected elements with bounding boxes and confidence scores
        """
        # Convert to grayscale if needed
        if len(screenshot.shape) == 3:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot

        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)  # type: ignore[call-overload]

        if not keypoints:
            return []

        # Cluster features into potential element regions
        regions = self._cluster_features(keypoints, screenshot.shape[:2])  # type: ignore[arg-type]

        # Convert regions to DetectedElement objects
        elements = []
        for region in regions:
            x, y, w, h = region["bounds"]
            feature_count = region["feature_count"]

            # Calculate confidence based on feature density
            area = w * h
            density = feature_count / area if area > 0 else 0
            confidence = min(1.0, density * 1000)  # Normalize

            if confidence >= self.confidence_threshold:
                # Crop the element image
                element_image = screenshot[y : y + h, x : x + w]

                elements.append(
                    DetectedElement(
                        element_type="feature_region",
                        bounds=(x, y, w, h),
                        confidence=confidence,
                        features={
                            "feature_count": feature_count,
                            "density": density,
                        },
                        image=element_image,
                    )
                )

        return elements

    def _cluster_features(self, keypoints: list, image_shape: tuple[int, int]) -> list[dict]:
        """Cluster nearby features into potential element regions.

        Args:
            keypoints: List of cv2.KeyPoint objects
            image_shape: (height, width) of image

        Returns:
            List of region dicts with bounds and feature_count
        """
        if not keypoints:
            return []

        # Extract keypoint coordinates
        points = np.array([kp.pt for kp in keypoints])

        # Simple grid-based clustering
        regions: list[dict] = []
        visited = set()

        for i, pt in enumerate(points):
            if i in visited:
                continue

            # Find nearby points
            cluster_points = [pt]
            cluster_indices = [i]

            for j, other_pt in enumerate(points):
                if j != i and j not in visited:
                    dist = np.sqrt((pt[0] - other_pt[0]) ** 2 + (pt[1] - other_pt[1]) ** 2)
                    if dist < self.cluster_distance:
                        cluster_points.append(other_pt)
                        cluster_indices.append(j)

            # Mark as visited
            for idx in cluster_indices:
                visited.add(idx)

            # Skip small clusters
            if len(cluster_points) < self.min_features:
                continue

            # Calculate bounding box
            cluster_array = np.array(cluster_points)
            x_min, y_min = cluster_array.min(axis=0).astype(int)
            x_max, y_max = cluster_array.max(axis=0).astype(int)

            # Add padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image_shape[1], x_max + padding)
            y_max = min(image_shape[0], y_max + padding)

            w, h = x_max - x_min, y_max - y_min

            # Filter by size
            if (
                w >= self.min_region_size[0]
                and h >= self.min_region_size[1]
                and w <= self.max_region_size[0]
                and h <= self.max_region_size[1]
            ):
                regions.append(
                    {
                        "bounds": (x_min, y_min, w, h),
                        "feature_count": len(cluster_points),
                    }
                )

        return regions

    def configure(self, **kwargs) -> None:
        """Configure feature detector parameters.

        Args:
            min_features: Minimum features to consider a region
            max_features: Maximum features per image
            min_region_size: Minimum (width, height) for regions
            max_region_size: Maximum (width, height) for regions
            cluster_distance: Distance threshold for clustering
            confidence_threshold: Minimum confidence for detections
        """
        if "min_features" in kwargs:
            self.min_features = kwargs["min_features"]
        if "max_features" in kwargs:
            self.max_features = kwargs["max_features"]
            self._orb = None  # Reset to apply new setting
        if "min_region_size" in kwargs:
            self.min_region_size = kwargs["min_region_size"]
        if "max_region_size" in kwargs:
            self.max_region_size = kwargs["max_region_size"]
        if "cluster_distance" in kwargs:
            self.cluster_distance = kwargs["cluster_distance"]
        if "confidence_threshold" in kwargs:
            self.confidence_threshold = kwargs["confidence_threshold"]


class OCRDetector(ElementDetector):
    """OCR-based text element detector.

    Detects and classifies text elements in screenshots for state construction.
    Uses EasyOCR for text detection and applies visual heuristics for classification.

    Text Classification (based on visual properties, not semantics):
    - heading: Large text (height > 24px) in upper portion of screen
    - label: Small text (height < 16px) typically near form fields
    - button_text: Text within button-like regions (detected by context)
    - paragraph: Multi-line text blocks
    - text: Default classification for other text

    This detector is used during state discovery to identify text elements
    that should become part of the state structure.
    """

    # Text element classification thresholds
    HEADING_MIN_HEIGHT = 24
    HEADING_SCREEN_POSITION = 0.25  # Top 25% of screen
    LABEL_MAX_HEIGHT = 16
    PARAGRAPH_MIN_WORDS = 10

    def __init__(self, languages: list[str] | None = None) -> None:
        """Initialize OCR detector.

        Args:
            languages: Language codes for OCR (default: ['en'])
        """
        self.languages = languages or ["en"]
        self.min_confidence = 0.5
        self.min_text_length = 1
        self._ocr_engine = None

    @property
    def ocr_engine(self):
        """Lazy-load OCR engine."""
        if self._ocr_engine is None:
            from qontinui.hal.implementations.easyocr_engine import EasyOCREngine

            self._ocr_engine = EasyOCREngine(languages=self.languages)
        return self._ocr_engine

    def detect(self, screenshot: np.ndarray) -> list[DetectedElement]:
        """Detect and classify text elements in screenshot.

        Args:
            screenshot: Screenshot image as numpy array (BGR)

        Returns:
            List of detected text elements with classification
        """
        # Convert numpy array to PIL Image for OCR engine
        if len(screenshot.shape) == 3:
            # BGR to RGB
            rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(rgb)
        image_height, image_width = screenshot.shape[:2]

        # Get text regions from OCR
        text_regions = self.ocr_engine.get_text_regions(
            pil_image, languages=self.languages, min_confidence=self.min_confidence
        )

        elements = []
        for region in text_regions:
            # Filter by text length
            if len(region.text.strip()) < self.min_text_length:
                continue

            # Classify the text element
            element_type = self._classify_text(region, image_height, image_width)

            # Crop the element image
            x, y, w, h = region.bounds
            element_image = screenshot[y : y + h, x : x + w] if h > 0 and w > 0 else None

            elements.append(
                DetectedElement(
                    element_type=element_type,
                    bounds=region.bounds,
                    confidence=region.confidence,
                    features={
                        "text": region.text,
                        "language": region.language,
                        "word_count": len(region.text.split()),
                    },
                    image=element_image,
                )
            )

        return elements

    def _classify_text(self, region, image_height: int, image_width: int) -> str:
        """Classify text element based on visual properties.

        Args:
            region: TextRegion from OCR
            image_height: Height of the source image
            image_width: Width of the source image

        Returns:
            Classification string: heading, label, paragraph, button_text, or text
        """
        x, y, w, h = region.bounds
        text = region.text.strip()
        word_count = len(text.split())

        # Check for heading (large text in upper portion)
        if h >= self.HEADING_MIN_HEIGHT:
            relative_y = y / image_height if image_height > 0 else 0
            if relative_y < self.HEADING_SCREEN_POSITION:
                return "heading"

        # Check for label (small text)
        if h <= self.LABEL_MAX_HEIGHT:
            return "label"

        # Check for paragraph (multi-line or many words)
        if word_count >= self.PARAGRAPH_MIN_WORDS:
            return "paragraph"

        # Check for button-like text (short, medium height, centered)
        if word_count <= 3 and self.LABEL_MAX_HEIGHT < h < self.HEADING_MIN_HEIGHT:
            # Could be button text - check aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if 1.5 < aspect_ratio < 10:  # Button-like aspect ratio
                return "button_text"

        return "text"

    def configure(self, **kwargs) -> None:
        """Configure OCR detector parameters.

        Args:
            languages: List of language codes for OCR
            min_confidence: Minimum OCR confidence threshold
            min_text_length: Minimum text length to include
            heading_min_height: Minimum height for heading classification
            label_max_height: Maximum height for label classification
        """
        if "languages" in kwargs:
            self.languages = kwargs["languages"]
            self._ocr_engine = None  # Reset to apply new languages
        if "min_confidence" in kwargs:
            self.min_confidence = kwargs["min_confidence"]
        if "min_text_length" in kwargs:
            self.min_text_length = kwargs["min_text_length"]
        if "heading_min_height" in kwargs:
            self.HEADING_MIN_HEIGHT = kwargs["heading_min_height"]
        if "label_max_height" in kwargs:
            self.LABEL_MAX_HEIGHT = kwargs["label_max_height"]
