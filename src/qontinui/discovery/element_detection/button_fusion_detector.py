"""
Button Fusion Detector - Multi-Strategy Fusion

Combines multiple detection signals (shape, text, color) with weighted voting
to achieve high recall from shape detection and high precision from text/color.
"""

import logging
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .analysis_base import (
    AnalysisInput,
    AnalysisResult,
    AnalysisType,
    BaseAnalyzer,
    BoundingBox,
    DetectedElement,
)

logger = logging.getLogger(__name__)


class ButtonFusionDetector(BaseAnalyzer):
    """
    Multi-strategy fusion detector for buttons

    Combines three complementary detection strategies:
    1. Shape Detection: High recall, finds rectangular regions with button-like properties
    2. Text Detection: High precision, validates buttons by detecting action text
    3. Color Detection: Validates through color distinctiveness

    Uses weighted voting to combine signals with adaptive confidence scoring.
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "button_fusion"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            # Strategy weights
            "shape_weight": 0.4,
            "text_weight": 0.4,
            "color_weight": 0.2,
            # Fusion parameters
            "min_agreement_score": 0.5,  # Minimum weighted agreement to detect
            "high_confidence_threshold": 0.75,  # When to mark as high confidence
            # Shape detection params
            "min_area": 400,
            "max_area": 50000,
            "min_aspect_ratio": 1.5,  # Buttons are typically wider than tall
            "max_aspect_ratio": 8.0,
            "edge_threshold": 50,
            # Text detection params
            "button_keywords": [
                "submit",
                "save",
                "cancel",
                "delete",
                "add",
                "create",
                "update",
                "confirm",
                "ok",
                "yes",
                "no",
                "login",
                "signup",
                "register",
                "search",
                "send",
                "upload",
                "download",
            ],
            # Color detection params
            "min_saturation": 30,
            "color_uniformity_threshold": 0.7,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Perform multi-strategy fusion detection"""
        logger.info(f"Running fusion detection on {len(input_data.screenshots)} screenshots")

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images = self._load_images(input_data.screenshot_data)

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, img in enumerate(images):
            elements = await self._analyze_screenshot(img, screenshot_idx, params)
            all_elements.extend(elements)

        # Calculate overall confidence based on agreement
        avg_confidence = np.mean([e.confidence for e in all_elements]) if all_elements else 0.0

        logger.info(
            f"Found {len(all_elements)} button candidates with "
            f"avg confidence {avg_confidence:.2f}"
        )

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=float(avg_confidence),
            metadata={
                "num_screenshots": len(images),
                "method": "multi_strategy_fusion",
                "strategy_weights": {
                    "shape": params["shape_weight"],
                    "text": params["text_weight"],
                    "color": params["color_weight"],
                },
                "parameters": params,
            },
        )

    def _load_images(self, screenshot_data: list[bytes]) -> list[np.ndarray]:
        """Load screenshots as numpy arrays"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(np.array(img))
        return images

    async def _analyze_screenshot(
        self, img: np.ndarray, screenshot_idx: int, params: dict[str, Any]
    ) -> list[DetectedElement]:
        """Analyze single screenshot using fusion approach"""

        # Run all three detection strategies
        shape_candidates = self._detect_by_shape(img, params)
        text_regions = self._detect_text_regions(img, params)
        color_regions = self._detect_by_color(img, params)

        logger.info(
            f"Screenshot {screenshot_idx}: "
            f"{len(shape_candidates)} shape, "
            f"{len(text_regions)} text, "
            f"{len(color_regions)} color candidates"
        )

        # Fuse detections using weighted voting
        fused_elements = self._fuse_detections(
            shape_candidates, text_regions, color_regions, params, screenshot_idx
        )

        return fused_elements

    def _detect_by_shape(
        self, img: np.ndarray, params: dict[str, Any]
    ) -> list[tuple[BoundingBox, float]]:
        """
        Detect button candidates by shape (rectangles with edges)
        Returns list of (bbox, confidence) tuples
        """
        candidates = []

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, params["edge_threshold"], params["edge_threshold"] * 2)

        # Morphological closing to connect edge fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Filter by size and aspect ratio
            if not (params["min_area"] <= area <= params["max_area"]):
                continue
            if not (params["min_aspect_ratio"] <= aspect_ratio <= params["max_aspect_ratio"]):
                continue

            # Calculate rectangularity (how close to perfect rectangle)
            contour_area = cv2.contourArea(contour)
            rectangularity = contour_area / area if area > 0 else 0

            # Buttons tend to be rectangular
            if rectangularity < 0.7:
                continue

            # Confidence based on rectangularity and aspect ratio
            # Higher for more rectangular shapes with typical button aspect ratios
            confidence = min(0.9, rectangularity * 0.7 + 0.2)

            candidates.append((BoundingBox(x=x, y=y, width=w, height=h), confidence))

        return candidates

    def _detect_text_regions(
        self, img: np.ndarray, params: dict[str, Any]
    ) -> list[tuple[BoundingBox, float]]:
        """
        Detect regions containing button-like text
        Returns list of (bbox, confidence) tuples
        """
        regions = []

        # In production, use OCR (pytesseract, EasyOCR, etc.)
        # For now, detect text-like patterns

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Use MSER (Maximally Stable Extremal Regions) to find text regions
        mser = cv2.MSER_create(_min_area=50, _max_area=5000, _delta=5)

        regions_mser, _ = mser.detectRegions(gray)

        for region in regions_mser:
            if len(region) < 10:
                continue

            x, y, w, h = cv2.boundingRect(region)

            # Text regions are typically small and horizontal
            aspect_ratio = w / h if h > 0 else 0
            if not (1.5 <= aspect_ratio <= 6.0):
                continue

            area = w * h
            if not (100 <= area <= 5000):
                continue

            # Add padding for button boundaries
            padding = 8
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(img.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(img.shape[0] - y_pad, h + 2 * padding)

            # Higher confidence for text in expected size range
            confidence = 0.7 if (200 <= area <= 2000) else 0.5

            regions.append((BoundingBox(x=x_pad, y=y_pad, width=w_pad, height=h_pad), confidence))

        return regions

    def _detect_by_color(
        self, img: np.ndarray, params: dict[str, Any]
    ) -> list[tuple[BoundingBox, float]]:
        """
        Detect buttons by color properties (uniform colored regions)
        Returns list of (bbox, confidence) tuples
        """
        regions = []

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Create mask for saturated colors (buttons often have distinct colors)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Buttons are typically saturated and not too dark/bright
        color_mask = (saturation > params["min_saturation"]) & (value > 50) & (value < 240)

        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color_mask = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            if not (400 <= area <= 50000):
                continue

            # Check color uniformity
            region_hsv = hsv[y : y + h, x : x + w]
            hue_std = np.std(region_hsv[:, :, 0])

            # Uniform colors suggest buttons
            uniformity = max(0, 1 - (hue_std / 180))

            if uniformity < params["color_uniformity_threshold"]:
                continue

            confidence = min(0.85, uniformity * 0.7 + 0.15)

            regions.append((BoundingBox(x=x, y=y, width=w, height=h), confidence))

        return regions

    def _fuse_detections(
        self,
        shape_candidates: list[tuple[BoundingBox, float]],
        text_regions: list[tuple[BoundingBox, float]],
        color_regions: list[tuple[BoundingBox, float]],
        params: dict[str, Any],
        screenshot_idx: int,
    ) -> list[DetectedElement]:
        """
        Fuse detections from multiple strategies using weighted voting
        """
        elements = []

        # Combine all candidates
        all_candidates = []

        for bbox, conf in shape_candidates:
            all_candidates.append(
                {
                    "bbox": bbox,
                    "votes": {"shape": conf * params["shape_weight"]},
                }
            )

        # Merge with text regions
        for bbox, conf in text_regions:
            merged = False
            for candidate in all_candidates:
                if self._boxes_overlap(candidate["bbox"], bbox, threshold=0.3):
                    candidate["votes"]["text"] = conf * params["text_weight"]
                    merged = True
                    break

            if not merged:
                all_candidates.append(
                    {
                        "bbox": bbox,
                        "votes": {"text": conf * params["text_weight"]},
                    }
                )

        # Merge with color regions
        for bbox, conf in color_regions:
            merged = False
            for candidate in all_candidates:
                if self._boxes_overlap(candidate["bbox"], bbox, threshold=0.3):
                    candidate["votes"]["color"] = conf * params["color_weight"]
                    merged = True
                    break

            if not merged:
                all_candidates.append(
                    {
                        "bbox": bbox,
                        "votes": {"color": conf * params["color_weight"]},
                    }
                )

        # Calculate final scores
        for candidate in all_candidates:
            votes = candidate["votes"]
            total_score = sum(votes.values())
            num_strategies = len(votes)

            # Require minimum agreement
            if total_score < params["min_agreement_score"]:
                continue

            # Bonus for multi-strategy agreement
            agreement_bonus = 0.1 * (num_strategies - 1)
            final_confidence = min(0.95, total_score + agreement_bonus)

            # Determine if high confidence
            is_high_confidence = final_confidence >= params["high_confidence_threshold"]

            elements.append(
                DetectedElement(
                    bounding_box=candidate["bbox"],
                    confidence=final_confidence,
                    label=("Button (Fused)" if is_high_confidence else "Button Candidate"),
                    element_type="button",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "fusion",
                        "strategies_agreed": num_strategies,
                        "votes": votes,
                        "total_score": total_score,
                        "high_confidence": is_high_confidence,
                    },
                )
            )

        return elements

    def _boxes_overlap(self, box1: BoundingBox, box2: BoundingBox, threshold: float = 0.3) -> bool:
        """Check if two bounding boxes overlap significantly"""
        return box1.iou(box2) >= threshold
