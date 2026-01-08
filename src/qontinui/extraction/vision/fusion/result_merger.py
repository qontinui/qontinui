"""
Result Fusion for Vision Extraction.

Merges StateImage candidates from multiple detection techniques
(edge detection, SAM3, OCR) using IoU-based deduplication.
"""

import logging

from ..models import (
    EdgeDetectionResult,
    ExtractedStateImageCandidate,
    FusionConfig,
    OCRResult,
    SAM3SegmentResult,
)

logger = logging.getLogger(__name__)


class ResultMerger:
    """
    Merge and deduplicate results from multiple detection techniques.

    Uses Intersection over Union (IoU) to identify overlapping detections
    and merges them into unified StateImage candidates.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        """
        Initialize the result merger.

        Args:
            config: Fusion configuration. Uses defaults if not provided.
        """
        self.config = config or FusionConfig()

    def merge(
        self,
        edge_results: list[EdgeDetectionResult] | None,
        sam_results: list[SAM3SegmentResult] | None,
        ocr_results: list[OCRResult] | None,
        screenshot_id: str,
    ) -> list[ExtractedStateImageCandidate]:
        """
        Merge results from all detection techniques.

        Args:
            edge_results: Results from edge detection.
            sam_results: Results from SAM3 segmentation.
            ocr_results: Results from OCR detection.
            screenshot_id: ID of the source screenshot.

        Returns:
            Deduplicated list of ExtractedStateImageCandidate.
        """
        logger.info("Merging detection results...")

        # Convert all results to candidates
        all_candidates: list[ExtractedStateImageCandidate] = []

        if edge_results:
            edge_candidates = self._edge_to_candidates(edge_results, screenshot_id)
            all_candidates.extend(edge_candidates)
            logger.debug(f"Added {len(edge_candidates)} candidates from edge detection")

        if sam_results:
            sam_candidates = self._sam_to_candidates(sam_results, screenshot_id)
            all_candidates.extend(sam_candidates)
            logger.debug(f"Added {len(sam_candidates)} candidates from SAM")

        if ocr_results:
            ocr_candidates = self._ocr_to_candidates(ocr_results, screenshot_id)
            all_candidates.extend(ocr_candidates)
            logger.debug(f"Added {len(ocr_candidates)} candidates from OCR")

        logger.info(f"Total candidates before deduplication: {len(all_candidates)}")

        # Deduplicate using IoU
        deduplicated = self._deduplicate_by_iou(all_candidates)

        # Limit to max candidates
        if len(deduplicated) > self.config.max_candidates:
            deduplicated = sorted(deduplicated, key=lambda c: c.confidence, reverse=True)[
                : self.config.max_candidates
            ]

        logger.info(f"Final candidates after deduplication: {len(deduplicated)}")

        return deduplicated

    def _edge_to_candidates(
        self,
        results: list[EdgeDetectionResult],
        screenshot_id: str,
    ) -> list[ExtractedStateImageCandidate]:
        """Convert edge detection results to candidates."""
        candidates = []

        for result in results:
            category = self._classify_edge_shape(result)

            candidate = ExtractedStateImageCandidate(
                id=f"edge_{result.id}",
                bbox=result.bbox,
                confidence=result.confidence,
                screenshot_id=screenshot_id,
                category=category,
                detection_technique="edge",
                is_clickable=(category in ("button", "input", "link")),
                metadata={
                    "contour_area": result.contour_area,
                    "vertex_count": result.vertex_count,
                    "aspect_ratio": result.aspect_ratio,
                    "source_technique": "edge",
                },
            )
            candidates.append(candidate)

        return candidates

    def _sam_to_candidates(
        self,
        results: list[SAM3SegmentResult],
        screenshot_id: str,
    ) -> list[ExtractedStateImageCandidate]:
        """Convert SAM results to candidates."""
        candidates = []

        for result in results:
            category = self._classify_sam_segment(result)

            candidate = ExtractedStateImageCandidate(
                id=f"sam_{result.id}",
                bbox=result.bbox,
                confidence=result.confidence,
                screenshot_id=screenshot_id,
                category=category,
                detection_technique="sam3",
                is_clickable=(category in ("button", "icon", "link")),
                metadata={
                    "mask_area": result.mask_area,
                    "stability_score": result.stability_score,
                    "predicted_iou": result.predicted_iou,
                    "source_technique": "sam3",
                },
            )
            candidates.append(candidate)

        return candidates

    def _ocr_to_candidates(
        self,
        results: list[OCRResult],
        screenshot_id: str,
    ) -> list[ExtractedStateImageCandidate]:
        """Convert OCR results to candidates."""
        candidates = []

        for result in results:
            category = self._classify_ocr_text(result)

            candidate = ExtractedStateImageCandidate(
                id=f"ocr_{result.id}",
                bbox=result.bbox,
                confidence=result.confidence,
                screenshot_id=screenshot_id,
                category=category,
                text=result.text,
                detection_technique="ocr",
                is_clickable=(category in ("button", "link")),
                metadata={
                    "language": result.language,
                    "text_length": len(result.text),
                    "source_technique": "ocr",
                },
            )
            candidates.append(candidate)

        return candidates

    def _classify_edge_shape(self, result: EdgeDetectionResult) -> str:
        """Classify edge detection result."""
        bbox = result.bbox
        aspect_ratio = result.aspect_ratio
        vertex_count = result.vertex_count

        if vertex_count == 4:
            if 1.5 < aspect_ratio < 6.0 and 30 < bbox.width < 300 and 20 < bbox.height < 60:
                return "button"
            if aspect_ratio > 4.0 and bbox.height < 50:
                return "input"
            if bbox.width > 200 and bbox.height > 100:
                return "container"

        if 0.8 < aspect_ratio < 1.25 and bbox.width < 60 and bbox.height < 60:
            return "icon"

        if result.contour_area > 10000:
            return "container"

        return "element"

    def _classify_sam_segment(self, result: SAM3SegmentResult) -> str:
        """Classify SAM segment result."""
        bbox = result.bbox
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0
        area = result.mask_area

        if area < 2500 and 0.7 < aspect_ratio < 1.4:
            return "icon"

        if 1.5 < aspect_ratio < 6.0 and 500 < area < 15000:
            return "button"

        if aspect_ratio > 4.0 and bbox.height < 50:
            return "input"

        if area > 50000:
            return "container"

        return "element"

    def _classify_ocr_text(self, result: OCRResult) -> str:
        """Classify OCR text result."""
        text = result.text.lower().strip()
        bbox = result.bbox

        button_keywords = [
            "submit",
            "cancel",
            "ok",
            "yes",
            "no",
            "save",
            "delete",
            "add",
            "remove",
            "edit",
            "update",
            "create",
            "close",
            "next",
            "back",
            "previous",
            "continue",
            "done",
            "finish",
            "login",
            "logout",
            "sign in",
            "sign out",
            "sign up",
            "search",
            "filter",
            "sort",
            "reset",
            "clear",
        ]

        if any(keyword in text for keyword in button_keywords):
            return "button"

        if text.startswith("http") or "click" in text or "learn more" in text:
            return "link"

        if len(text) < 15 and 1.5 < bbox.width / max(bbox.height, 1) < 6:
            return "button"

        if len(text) > 50:
            return "paragraph"

        return "label"

    def _deduplicate_by_iou(
        self,
        candidates: list[ExtractedStateImageCandidate],
    ) -> list[ExtractedStateImageCandidate]:
        """
        Remove duplicate/overlapping candidates using IoU.

        When candidates overlap above the IoU threshold, keeps the one
        with higher confidence and merges metadata.
        """
        if not candidates:
            return []

        # Sort by confidence (descending) to prefer higher confidence
        if self.config.prefer_higher_confidence:
            sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        else:
            sorted_candidates = list(candidates)

        kept: list[ExtractedStateImageCandidate] = []

        for candidate in sorted_candidates:
            # Check if this candidate overlaps with any kept candidate
            merged = False

            for idx, kept_candidate in enumerate(kept):
                iou = candidate.bbox.iou(kept_candidate.bbox)

                if iou > self.config.iou_threshold:
                    # Merge: keep higher confidence, combine metadata
                    if candidate.confidence > kept_candidate.confidence:
                        # Replace with higher confidence candidate
                        merged_candidate = self._merge_candidates(candidate, kept_candidate)
                        kept[idx] = merged_candidate
                    else:
                        # Keep existing, add metadata from new
                        merged_candidate = self._merge_candidates(kept_candidate, candidate)
                        kept[idx] = merged_candidate

                    merged = True
                    break

            if not merged:
                kept.append(candidate)

        return kept

    def _merge_candidates(
        self,
        primary: ExtractedStateImageCandidate,
        secondary: ExtractedStateImageCandidate,
    ) -> ExtractedStateImageCandidate:
        """
        Merge two overlapping candidates.

        Keeps primary's core attributes but combines metadata and
        picks the best text/description from either.
        """
        # Combine metadata
        merged_metadata = {**secondary.metadata, **primary.metadata}
        merged_metadata["merged_from"] = [primary.id, secondary.id]
        merged_metadata["detection_techniques"] = list(
            {
                primary.detection_technique,
                secondary.detection_technique,
            }
        )

        # Pick text if primary doesn't have it
        text = primary.text or secondary.text

        # Pick description if primary doesn't have it
        description = primary.description or secondary.description

        # Combine is_clickable (true if either says clickable)
        is_clickable = primary.is_clickable or secondary.is_clickable

        return ExtractedStateImageCandidate(
            id=primary.id,
            bbox=primary.bbox,
            confidence=primary.confidence,
            screenshot_id=primary.screenshot_id,
            cropped_image_path=primary.cropped_image_path or secondary.cropped_image_path,
            category=primary.category,
            text=text,
            description=description,
            extraction_method=primary.extraction_method,
            detection_technique=f"{primary.detection_technique}+{secondary.detection_technique}",
            source_url=primary.source_url or secondary.source_url,
            is_clickable=is_clickable,
            metadata=merged_metadata,
        )
