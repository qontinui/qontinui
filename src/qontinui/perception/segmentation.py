"""Screen segmentation for UI element detection."""

from typing import Any

import cv2
import numpy as np


class ScreenSegmenter:
    """Segment screenshots into UI elements using various methods."""

    def __init__(self, use_sam: bool = False, sam_checkpoint: str | None = None):
        """Initialize ScreenSegmenter.

        Args:
            use_sam: Whether to use Segment Anything Model
            sam_checkpoint: Path to SAM checkpoint file
        """
        self.use_sam = use_sam
        self.sam = None
        self.mask_generator = None

        if use_sam:
            try:
                from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

                # Default to base model if no checkpoint specified
                model_type = "vit_b"
                if sam_checkpoint:
                    # Infer model type from checkpoint name
                    if "vit_h" in sam_checkpoint:
                        model_type = "vit_h"
                    elif "vit_l" in sam_checkpoint:
                        model_type = "vit_l"

                self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                self.mask_generator = SamAutomaticMaskGenerator(
                    self.sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
                print(f"SAM initialized with {model_type} model")
            except ImportError:
                print("SAM not available, falling back to OpenCV segmentation")
                self.use_sam = False
            except Exception as e:
                print(f"Failed to initialize SAM: {e}, falling back to OpenCV")
                self.use_sam = False

    def segment_screen(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Segment a screenshot into UI elements.

        Args:
            screenshot: Screenshot as numpy array (BGR format)

        Returns:
            List of segments with bbox and image data
        """
        if self.use_sam and self.mask_generator:
            return self._segment_with_sam(screenshot)
        else:
            return self._segment_with_opencv(screenshot)

    def _segment_with_sam(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Segment using Segment Anything Model.

        Args:
            screenshot: Screenshot as numpy array

        Returns:
            List of segments
        """
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

        # Generate masks
        masks = self.mask_generator.generate(image_rgb)

        segments = []
        for i, mask_data in enumerate(masks):
            # Extract bounding box
            bbox = mask_data["bbox"]  # [x, y, w, h] format
            x, y, w, h = (int(v) for v in bbox)

            # Crop image region
            cropped = self._crop_image(screenshot, (x, y, w, h))

            # Store segment info
            segment = {
                "id": f"sam_segment_{i}",
                "bbox": (x, y, w, h),
                "image": cropped,
                "area": mask_data.get("area", w * h),
                "predicted_iou": mask_data.get("predicted_iou", 0.0),
                "stability_score": mask_data.get("stability_score", 0.0),
                "mask": mask_data.get("segmentation", None),
            }
            segments.append(segment)

        return segments

    def _segment_with_opencv(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Segment using OpenCV contour detection.

        Args:
            screenshot: Screenshot as numpy array

        Returns:
            List of segments
        """
        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Morphological operations to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segments = []
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out very small regions
            if w < 10 or h < 10:
                continue

            # Crop image region
            cropped = self._crop_image(screenshot, (x, y, w, h))

            # Calculate contour area
            area = cv2.contourArea(contour)

            segment = {
                "id": f"opencv_segment_{i}",
                "bbox": (x, y, w, h),
                "image": cropped,
                "area": area,
                "contour": contour,
            }
            segments.append(segment)

        # Sort by area (largest first)
        segments.sort(key=lambda s: s["area"], reverse=True)

        return segments

    def _crop_image(self, img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Crop image to bounding box.

        Args:
            img: Source image
            bbox: Bounding box (x, y, width, height)

        Returns:
            Cropped image
        """
        x, y, w, h = bbox
        # Ensure bounds are within image
        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        x2 = min(x + w, w_img)
        y2 = min(y + h, h_img)

        return img[y:y2, x:x2]

    def detect_text_regions(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Detect text regions in screenshot.

        Args:
            screenshot: Screenshot as numpy array

        Returns:
            List of text regions with bounding boxes
        """
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on aspect ratio (text is usually wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 1.5 and w > 20:
                cropped = self._crop_image(screenshot, (x, y, w, h))
                text_regions.append(
                    {
                        "id": f"text_region_{i}",
                        "bbox": (x, y, w, h),
                        "image": cropped,
                        "aspect_ratio": aspect_ratio,
                    }
                )

        return text_regions

    def detect_buttons(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Detect button-like elements in screenshot.

        Args:
            screenshot: Screenshot as numpy array

        Returns:
            List of potential buttons
        """
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Detect rectangles using edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buttons = []
        for i, contour in enumerate(contours):
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if shape is roughly rectangular (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size and aspect ratio typical for buttons
                if 20 < w < 300 and 15 < h < 100:
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 8:
                        cropped = self._crop_image(screenshot, (x, y, w, h))
                        buttons.append(
                            {
                                "id": f"button_{i}",
                                "bbox": (x, y, w, h),
                                "image": cropped,
                                "aspect_ratio": aspect_ratio,
                            }
                        )

        return buttons
