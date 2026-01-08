"""
SAM3 Segmentation for StateImage Candidate Discovery.

Uses Segment Anything Model (SAM) for precise UI element boundary detection.
Supports both the original SAM (segment_anything) and SAM3.
"""

import base64
import logging
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..models import (
    BoundingBox,
    ExtractedStateImageCandidate,
    SAM3Config,
    SAM3SegmentResult,
)

logger = logging.getLogger(__name__)


class SAM3Segmenter:
    """
    Segment Anything Model integration for UI element detection.

    Uses SAM's automatic mask generation to find UI element boundaries.
    Falls back to SAM2/SAM if SAM3 is not available.
    """

    def __init__(self, config: SAM3Config | None = None) -> None:
        """
        Initialize the SAM3 segmenter.

        Args:
            config: SAM3 configuration. Uses defaults if not provided.
        """
        self.config = config or SAM3Config()
        self._sam_model = None
        self._mask_generator = None
        self._sam_available = False
        self._sam_version = "none"
        self._result_counter = 0

        if self.config.enabled:
            self._try_load_sam()

    def _try_load_sam(self) -> None:
        """Try to load SAM model (SAM3 first, then SAM2, then SAM)."""
        # Try SAM3 first
        if self._try_load_sam3():
            return

        # Fall back to original SAM
        if self._try_load_sam_original():
            return

        logger.warning("No SAM model available. Segmentation will be skipped.")

    def _try_load_sam3(self) -> bool:
        """Try to load SAM3 model."""
        try:
            # Check for checkpoint
            import os

            import torch
            from sam3.build_sam import build_sam3_image_model

            checkpoint_paths = [
                "checkpoints/sam3_hiera_large.pt",
                os.path.expanduser("~/.cache/sam3/sam3_hiera_large.pt"),
            ]

            checkpoint = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint = path
                    break

            if checkpoint is None:
                logger.debug("SAM3 checkpoint not found")
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._sam_model = build_sam3_image_model(checkpoint, device=device)

            from sam3 import Sam3Processor

            self._mask_generator = Sam3Processor(self._sam_model)
            self._sam_available = True
            self._sam_version = "sam3"
            logger.info(f"SAM3 loaded on {device}")
            return True

        except ImportError:
            logger.debug("SAM3 not installed")
            return False
        except Exception as e:
            logger.debug(f"Failed to load SAM3: {e}")
            return False

    def _try_load_sam_original(self) -> bool:
        """Try to load original SAM model."""
        try:
            import os

            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            # Find checkpoint
            checkpoint_paths = [
                f"checkpoints/sam_{self.config.model_type}.pth",
                os.path.expanduser(f"~/.cache/sam/sam_{self.config.model_type}.pth"),
            ]

            checkpoint = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint = path
                    break

            if checkpoint is None:
                logger.debug("SAM checkpoint not found")
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._sam_model = sam_model_registry[self.config.model_type](checkpoint=checkpoint)
            self._sam_model.to(device)

            self._mask_generator = SamAutomaticMaskGenerator(
                self._sam_model,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                min_mask_region_area=self.config.min_mask_region_area,
            )

            self._sam_available = True
            self._sam_version = "sam"
            logger.info(f"SAM ({self.config.model_type}) loaded on {device}")
            return True

        except ImportError:
            logger.debug("segment_anything not installed")
            return False
        except Exception as e:
            logger.debug(f"Failed to load SAM: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if SAM is available."""
        return self._sam_available

    def segment(
        self,
        screenshot: np.ndarray,
        screenshot_id: str,
    ) -> tuple[list[SAM3SegmentResult], np.ndarray | None]:
        """
        Segment screenshot using SAM.

        Args:
            screenshot: BGR image as numpy array.
            screenshot_id: ID of the screenshot for reference.

        Returns:
            Tuple of:
                - List of SAM3SegmentResult
                - Mask overlay image (debug visualization)
        """
        if not self.config.enabled:
            return [], None

        if not self._sam_available:
            logger.warning("SAM not available, skipping segmentation")
            return [], None

        logger.info(f"Running SAM segmentation ({self._sam_version})...")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

        try:
            if self._sam_version == "sam3":
                masks_data = self._segment_with_sam3(image_rgb)
            else:
                masks_data = self._segment_with_sam(image_rgb)

            # Process masks into results
            results: list[SAM3SegmentResult] = []

            for mask_data in masks_data:
                bbox = mask_data.get("bbox")
                if bbox is None:
                    continue

                x, y, w, h = (int(v) for v in bbox)
                area = mask_data.get("area", w * h)

                # Filter by area
                if area < self.config.min_mask_region_area:
                    continue
                if area > self.config.max_mask_region_area:
                    continue

                self._result_counter += 1
                result_id = f"sam_{self._result_counter:06d}"

                # Get mask RLE for debug visualization
                mask = mask_data.get("segmentation")
                mask_rle = self._encode_mask_rle(mask) if mask is not None else None

                result = SAM3SegmentResult(
                    id=result_id,
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    mask_area=int(area),
                    stability_score=float(mask_data.get("stability_score", 0.0)),
                    predicted_iou=float(mask_data.get("predicted_iou", 0.0)),
                    confidence=float(mask_data.get("stability_score", 0.0)),
                    mask_rle=mask_rle,
                )
                results.append(result)

            # Create debug overlay
            overlay = self._create_mask_overlay(screenshot, masks_data)

            logger.info(f"SAM segmentation found {len(results)} segments")

            return results, overlay

        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}")
            return [], None

    def _segment_with_sam3(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Segment using SAM3."""
        pil_image = Image.fromarray(image_rgb)
        self._mask_generator.set_image(pil_image)

        # Automatic segmentation using grid points
        h, w = image_rgb.shape[:2]
        grid_points = self.config.points_per_side
        step_x = w // grid_points
        step_y = h // grid_points

        all_masks: list[dict[str, Any]] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()

        for i in range(grid_points):
            for j in range(grid_points):
                x = step_x * i + step_x // 2
                y = step_y * j + step_y // 2

                try:
                    results = self._mask_generator.segment_from_point(x, y)

                    if results and "masks" in results and len(results["masks"]) > 0:
                        mask = results["masks"][0]

                        # Get bounding box from mask
                        y_indices, x_indices = np.where(mask)
                        if len(x_indices) == 0:
                            continue

                        x1, x2 = int(x_indices.min()), int(x_indices.max())
                        y1, y2 = int(y_indices.min()), int(y_indices.max())

                        # Deduplicate by bounding box
                        box_key = (x1, y1, x2, y2)
                        if box_key in seen_boxes:
                            continue
                        seen_boxes.add(box_key)

                        mask_data = {
                            "bbox": (x1, y1, x2 - x1, y2 - y1),
                            "area": int(mask.sum()),
                            "segmentation": mask,
                            "stability_score": results.get("scores", [0.9])[0],
                            "predicted_iou": results.get("scores", [0.9])[0],
                        }
                        all_masks.append(mask_data)

                except Exception:
                    continue

        return all_masks

    def _segment_with_sam(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Segment using original SAM."""
        masks = self._mask_generator.generate(image_rgb)
        return masks

    def _encode_mask_rle(self, mask: np.ndarray | None) -> dict[str, Any] | None:
        """Encode mask as run-length encoding for transport."""
        if mask is None:
            return None

        # Simple RLE encoding
        flat = mask.flatten()
        runs = []
        start = 0
        current_val = flat[0]

        for i, val in enumerate(flat):
            if val != current_val:
                runs.append({"start": start, "length": i - start, "value": int(current_val)})
                start = i
                current_val = val

        runs.append({"start": start, "length": len(flat) - start, "value": int(current_val)})

        return {
            "shape": list(mask.shape),
            "runs": runs,
        }

    def _create_mask_overlay(
        self,
        screenshot: np.ndarray,
        masks_data: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Create debug visualization with colored masks overlaid.

        Args:
            screenshot: Original screenshot.
            masks_data: List of mask data from SAM.

        Returns:
            BGR image with mask overlays.
        """
        overlay = screenshot.copy()

        # Generate colors for masks
        colors = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 128, 255),  # Light red
            (255, 128, 128),  # Light blue
        ]

        for idx, mask_data in enumerate(masks_data):
            mask = mask_data.get("segmentation")
            if mask is None:
                continue

            color = colors[idx % len(colors)]

            # Create colored mask overlay
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask] = color

            # Blend with original
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.3, 0)

            # Draw bounding box
            bbox = mask_data.get("bbox")
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        return overlay

    def results_to_candidates(
        self,
        sam_results: list[SAM3SegmentResult],
        screenshot_id: str,
    ) -> list[ExtractedStateImageCandidate]:
        """
        Convert SAM results to StateImage candidates.

        Args:
            sam_results: Results from SAM segmentation.
            screenshot_id: ID of the source screenshot.

        Returns:
            List of ExtractedStateImageCandidate.
        """
        candidates = []

        for result in sam_results:
            # Classify based on size and shape
            category = self._classify_segment(result)

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
                },
            )
            candidates.append(candidate)

        return candidates

    def _classify_segment(self, result: SAM3SegmentResult) -> str:
        """
        Classify segment as element category.

        This is for description only - categories don't have
        functional significance in the state machine.
        """
        bbox = result.bbox
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0
        area = result.mask_area

        # Small and roughly square: icon
        if area < 2500 and 0.7 < aspect_ratio < 1.4:
            return "icon"

        # Button-like: moderate size, rectangular
        if 1.5 < aspect_ratio < 6.0 and 500 < area < 15000:
            return "button"

        # Input field: wide and short
        if aspect_ratio > 4.0 and bbox.height < 50:
            return "input"

        # Large area: container
        if area > 50000:
            return "container"

        return "element"

    def get_overlay_base64(self, overlay: np.ndarray) -> str:
        """
        Convert overlay image to base64 for API transport.

        Args:
            overlay: BGR numpy array.

        Returns:
            Base64-encoded PNG string.
        """
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def reset_counter(self) -> None:
        """Reset the result counter (call between screenshots)."""
        self._result_counter = 0
