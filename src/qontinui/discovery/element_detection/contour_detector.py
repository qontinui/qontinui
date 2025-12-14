"""Contour-based UI element detection.

This module provides functionality for detecting UI elements using edge detection
and contour analysis. It can identify interactive elements like buttons, icons,
and other UI components based on their visual boundaries.

Key Features:
- Multiple edge detection methods (Canny, Sobel, Laplacian)
- Configurable contour filtering by size and area
- Integration with existing DetectedElement model
- Support for extracting element images

Example:
    >>> from qontinui.discovery.element_detection import ContourElementDetector
    >>> detector = ContourElementDetector()
    >>> elements = detector.detect(screenshot)
    >>> for element in elements:
    ...     print(f"Found element at {element.bounds}")
"""

import logging
from typing import Any

import cv2
import numpy as np

from .detector import DetectedElement, ElementDetector

logger = logging.getLogger(__name__)


class ContourElementDetector(ElementDetector):
    """Detects UI elements using contour analysis.

    This detector uses edge detection algorithms to find boundaries of UI elements
    in screenshots. It can identify buttons, icons, and other interactive elements
    based on their visual contours.

    The detector:
    1. Applies edge detection (Canny, Sobel, or Laplacian)
    2. Finds contours in the edge map
    3. Filters contours by area and size
    4. Returns bounding boxes of detected elements
    """

    def __init__(
        self,
        edge_method: str = "canny",
        min_area: int = 100,
        min_size: tuple[int, int] = (20, 20),
        max_size: tuple[int, int] = (500, 500),
        max_contours: int = 50,
    ) -> None:
        """Initialize the contour detector.

        Args:
            edge_method: Edge detection method ("canny", "sobel", "laplacian").
            min_area: Minimum contour area in pixels.
            min_size: Minimum (width, height) for detected elements.
            max_size: Maximum (width, height) for detected elements.
            max_contours: Maximum number of contours to process.
        """
        self.edge_method = edge_method
        self.min_area = min_area
        self.min_size = min_size
        self.max_size = max_size
        self.max_contours = max_contours

        # Edge detection parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150

    def detect(self, screenshot: np.ndarray[Any, Any]) -> list[DetectedElement]:
        """Detect UI elements using contour analysis.

        Args:
            screenshot: Screenshot image as numpy array (BGR or grayscale).

        Returns:
            List of detected elements with bounding boxes.
        """
        try:
            # Convert to grayscale
            if len(screenshot.shape) == 3:
                gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            else:
                gray = screenshot

            # Apply edge detection
            edges = self._detect_edges(gray)

            # Find contours
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # Convert to bounding boxes and filter
            elements = []
            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size
                if (
                    w < self.min_size[0]
                    or h < self.min_size[1]
                    or w > self.max_size[0]
                    or h > self.max_size[1]
                ):
                    continue

                # Calculate confidence based on contour properties
                bbox_area = w * h
                fill_ratio = area / bbox_area if bbox_area > 0 else 0
                confidence = min(0.95, 0.5 + fill_ratio * 0.45)

                # Extract element image
                element_image = screenshot[y : y + h, x : x + w].copy()

                elements.append(
                    DetectedElement(
                        element_type="contour_region",
                        bounds=(x, y, w, h),
                        confidence=confidence,
                        features={
                            "area": int(area),
                            "fill_ratio": float(fill_ratio),
                            "edge_method": self.edge_method,
                        },
                        image=element_image,
                    )
                )

                # Limit number of contours
                if len(elements) >= self.max_contours:
                    break

            logger.debug(f"Detected {len(elements)} contours using {self.edge_method}")
            return elements

        except Exception as e:
            logger.error(f"Error detecting contours: {e}")
            return []

    def _detect_edges(self, gray: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply edge detection to grayscale image.

        Args:
            gray: Grayscale image.

        Returns:
            Binary edge map.
        """
        if self.edge_method == "canny":
            edges = cv2.Canny(
                gray,
                self.canny_threshold1,
                self.canny_threshold2,
            )
        elif self.edge_method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            edges = magnitude.astype(np.uint8)
        elif self.edge_method == "laplacian":
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.absolute(laplacian).astype(np.uint8)
        else:
            logger.warning(f"Unknown edge detection method: {self.edge_method}")
            edges = cv2.Canny(gray, 50, 150)

        return edges  # type: ignore[no-any-return]

    def configure(self, **kwargs: Any) -> None:
        """Configure detector parameters.

        Args:
            edge_method: Edge detection method.
            min_area: Minimum contour area.
            min_size: Minimum element size.
            max_size: Maximum element size.
            max_contours: Maximum contours to process.
            canny_threshold1: Canny low threshold.
            canny_threshold2: Canny high threshold.
        """
        if "edge_method" in kwargs:
            self.edge_method = kwargs["edge_method"]
        if "min_area" in kwargs:
            self.min_area = kwargs["min_area"]
        if "min_size" in kwargs:
            self.min_size = kwargs["min_size"]
        if "max_size" in kwargs:
            self.max_size = kwargs["max_size"]
        if "max_contours" in kwargs:
            self.max_contours = kwargs["max_contours"]
        if "canny_threshold1" in kwargs:
            self.canny_threshold1 = kwargs["canny_threshold1"]
        if "canny_threshold2" in kwargs:
            self.canny_threshold2 = kwargs["canny_threshold2"]


class EdgeBasedCropper:
    """Refines region boundaries using edge detection.

    This class takes an initial region and finds the optimal crop by detecting
    edges within that region and finding the tightest bounding box around
    significant content.
    """

    def __init__(
        self,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        padding: int = 5,
    ) -> None:
        """Initialize the cropper.

        Args:
            canny_threshold1: Canny edge detection low threshold.
            canny_threshold2: Canny edge detection high threshold.
            padding: Padding to add around refined boundaries.
        """
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.padding = padding

    def find_best_crop(
        self,
        screenshot: np.ndarray[Any, Any],
        initial_region: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Find optimal crop boundaries using edge detection.

        Refines an initial region by detecting edges and finding the tightest
        bounding box around significant content.

        Args:
            screenshot: Screenshot image (BGR or grayscale).
            initial_region: Initial region as (x, y, width, height).

        Returns:
            Refined region as (x, y, width, height).
        """
        try:
            x, y, w, h = initial_region

            # Extract region
            roi = screenshot[y : y + h, x : x + w]

            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            # Apply edge detection
            edges = cv2.Canny(
                gray,
                self.canny_threshold1,
                self.canny_threshold2,
            )

            # Find contours in the ROI
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                # Get bounding box of all contours combined
                all_points = np.vstack(contours)
                rx, ry, rw, rh = cv2.boundingRect(all_points)

                # Add padding
                rx = max(0, rx - self.padding)
                ry = max(0, ry - self.padding)
                rw = min(w - rx, rw + 2 * self.padding)
                rh = min(h - ry, rh + 2 * self.padding)

                # Return refined region in original coordinates
                return (x + rx, y + ry, rw, rh)

            # If no contours found, return original region
            return initial_region

        except Exception as e:
            logger.error(f"Error finding best crop for region {initial_region}: {e}")
            return initial_region
