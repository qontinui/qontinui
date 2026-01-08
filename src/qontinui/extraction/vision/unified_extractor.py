"""
Unified Vision Extractor for StateImage Candidate Discovery.

Orchestrates multiple detection techniques (edge detection, SAM3, OCR)
to discover StateImage candidates from screenshots. Each technique's
results are preserved for debugging, then merged into unified candidates.
"""

import logging
import uuid
from pathlib import Path

import cv2
import numpy as np

from .edge import EdgeDetector
from .fusion import ResultMerger
from .models import (
    ScreenshotInfo,
    VisionExtractionConfig,
    VisionExtractionResult,
)
from .ocr import OCRDetector
from .sam3 import SAM3Segmenter

logger = logging.getLogger(__name__)


class UnifiedVisionExtractor:
    """
    Unified extractor that runs multiple vision detection techniques.

    Combines edge detection, SAM3 segmentation, and OCR to discover
    StateImage candidates. Each technique's results are preserved
    separately for debugging in qontinui-web.
    """

    def __init__(self, config: VisionExtractionConfig | None = None) -> None:
        """
        Initialize the unified vision extractor.

        Args:
            config: Extraction configuration. Uses defaults if not provided.
        """
        self.config = config or VisionExtractionConfig()

        # Initialize detection modules
        self.edge_detector = EdgeDetector(self.config.edge_detection)
        self.sam_segmenter = SAM3Segmenter(self.config.sam3)
        self.ocr_detector = OCRDetector(self.config.ocr)
        self.result_merger = ResultMerger(self.config.fusion)

    async def extract(
        self,
        screenshot_path: Path | str,
        screenshot_id: str | None = None,
        source_url: str | None = None,
    ) -> VisionExtractionResult:
        """
        Run all detection techniques and return unified results.

        Args:
            screenshot_path: Path to screenshot image.
            screenshot_id: Optional ID for the screenshot. Auto-generated if not provided.
            source_url: Optional source URL where screenshot was captured.

        Returns:
            VisionExtractionResult with candidates and debug data.
        """
        extraction_id = str(uuid.uuid4())
        screenshot_path = Path(screenshot_path)

        if screenshot_id is None:
            screenshot_id = f"screenshot_{extraction_id[:8]}"

        logger.info(f"Starting vision extraction: {extraction_id}")
        logger.info(f"Screenshot: {screenshot_path}")

        result = VisionExtractionResult(
            extraction_id=extraction_id,
            config=self.config.to_dict(),
        )

        # Load screenshot
        screenshot = self._load_screenshot(screenshot_path)
        if screenshot is None:
            result.add_error(f"Failed to load screenshot: {screenshot_path}")
            result.complete()
            return result

        # Record screenshot info
        height, width = screenshot.shape[:2]
        screenshot_info = ScreenshotInfo(
            id=screenshot_id,
            path=screenshot_path,
            width=width,
            height=height,
            source_url=source_url,
        )
        result.screenshots.append(screenshot_info)

        # Reset counters between screenshots
        self.edge_detector.reset_counter()
        self.ocr_detector.reset_counter()
        self.sam_segmenter.reset_counter()

        # Run edge detection
        try:
            edge_results, contour_results, edge_overlay = self.edge_detector.detect(
                screenshot, screenshot_id
            )
            result.edge_detection_results = edge_results
            result.contour_results = contour_results

            if edge_overlay is not None and self.config.save_debug_images:
                result.edge_overlay_image = self.edge_detector.get_overlay_base64(edge_overlay)
                result.contour_overlay_image = result.edge_overlay_image

        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            result.add_warning(f"Edge detection failed: {e}")
            edge_results = []

        # Run SAM3 segmentation
        try:
            sam_results, sam_overlay = self.sam_segmenter.segment(screenshot, screenshot_id)
            result.sam3_segments = sam_results

            if sam_overlay is not None and self.config.save_debug_images:
                result.sam3_mask_image = self.sam_segmenter.get_overlay_base64(sam_overlay)

        except Exception as e:
            logger.error(f"SAM3 segmentation failed: {e}")
            result.add_warning(f"SAM3 segmentation failed: {e}")
            sam_results = []

        # Run OCR detection
        try:
            ocr_results, ocr_overlay = self.ocr_detector.detect(screenshot, screenshot_id)
            result.ocr_results = ocr_results

            if ocr_overlay is not None and self.config.save_debug_images:
                result.ocr_overlay_image = self.ocr_detector.get_overlay_base64(ocr_overlay)

        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            result.add_warning(f"OCR detection failed: {e}")
            ocr_results = []

        # Merge results
        try:
            candidates = self.result_merger.merge(
                edge_results=result.edge_detection_results,
                sam_results=result.sam3_segments,
                ocr_results=result.ocr_results,
                screenshot_id=screenshot_id,
            )

            # Add source URL to all candidates
            if source_url:
                for candidate in candidates:
                    candidate.source_url = source_url

            result.candidates = candidates

        except Exception as e:
            logger.error(f"Result merging failed: {e}")
            result.add_error(f"Result merging failed: {e}")

        # Save cropped candidates if configured
        if self.config.save_cropped_candidates and self.config.output_dir:
            self._save_cropped_candidates(screenshot, result, screenshot_path)

        result.complete()

        logger.info(
            f"Vision extraction complete in {result.duration_ms:.1f}ms: "
            f"{len(result.candidates)} candidates"
        )

        return result

    def extract_sync(
        self,
        screenshot_path: Path | str,
        screenshot_id: str | None = None,
        source_url: str | None = None,
    ) -> VisionExtractionResult:
        """
        Synchronous version of extract.

        Args:
            screenshot_path: Path to screenshot image.
            screenshot_id: Optional ID for the screenshot.
            source_url: Optional source URL where screenshot was captured.

        Returns:
            VisionExtractionResult with candidates and debug data.
        """
        import asyncio

        return asyncio.run(self.extract(screenshot_path, screenshot_id, source_url))

    def _load_screenshot(self, screenshot_path: Path) -> np.ndarray | None:
        """Load screenshot from path."""
        if not screenshot_path.exists():
            logger.error(f"Screenshot not found: {screenshot_path}")
            return None

        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            logger.error(f"Failed to read screenshot: {screenshot_path}")
            return None

        return screenshot

    def _save_cropped_candidates(
        self,
        screenshot: np.ndarray,
        result: VisionExtractionResult,
        screenshot_path: Path,
    ) -> None:
        """Save cropped images for each candidate."""
        if not self.config.output_dir:
            return

        output_dir = Path(self.config.output_dir)
        crops_dir = output_dir / "crops" / screenshot_path.stem
        crops_dir.mkdir(parents=True, exist_ok=True)

        height, width = screenshot.shape[:2]

        for candidate in result.candidates:
            bbox = candidate.bbox

            # Ensure bounds are within image
            x1 = max(0, bbox.x)
            y1 = max(0, bbox.y)
            x2 = min(width, bbox.x2)
            y2 = min(height, bbox.y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop and save
            cropped = screenshot[y1:y2, x1:x2]
            crop_path = crops_dir / f"{candidate.id}.png"

            cv2.imwrite(str(crop_path), cropped)
            candidate.cropped_image_path = crop_path

    def get_available_techniques(self) -> dict[str, bool]:
        """
        Check which detection techniques are available.

        Returns:
            Dict mapping technique name to availability.
        """
        return {
            "edge_detection": self.config.edge_detection.enabled,
            "sam3": self.sam_segmenter.is_available,
            "ocr": self.ocr_detector.is_available,
        }
