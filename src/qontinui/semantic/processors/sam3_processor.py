"""SAM3 processor for semantic segmentation."""

import time
from typing import Any

import numpy as np

try:
    # Try importing SAM3
    from sam3 import Sam3Processor as Sam3ProcessorCore
    from sam3.build_sam import build_sam3_image_model

    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False

from ..core import PixelLocation, SemanticObject, SemanticScene
from ..core.semantic_object import ObjectType
from ..description import BasicDescriptionGenerator, DescriptionGenerator
from .base import SemanticProcessor


class SAM3Processor(SemanticProcessor):
    """Semantic processor using SAM3 for pixel-level segmentation.

    SAM3 (Segment Anything Model 3) provides pixel-perfect segmentation masks
    for objects in images with support for text-based concept prompting.
    This processor generates semantic descriptions for the segmented regions
    and can optionally apply OCR to extract text.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        description_generator: DescriptionGenerator | None = None,
    ) -> None:
        """Initialize SAM3 processor.

        Args:
            checkpoint_path: Path to model checkpoint (if None, uses default)
            description_generator: Generator for creating semantic descriptions
        """
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.sam3_model = None
        self.processor = None

        # Use provided generator or fall back to basic
        self.description_generator = description_generator or BasicDescriptionGenerator()

        if HAS_SAM3:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize SAM3 model."""
        if not HAS_SAM3:
            return

        # Determine device
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load SAM3 model
        self.sam3_model = build_sam3_image_model(self.checkpoint_path, device=device)

        # Create processor for handling images and prompts
        self.processor = Sam3ProcessorCore(self.sam3_model)

    def set_description_generator(self, generator: DescriptionGenerator) -> None:
        """Set the description generator for segments.

        Args:
            generator: DescriptionGenerator instance for creating semantic descriptions
        """
        self.description_generator = generator

    def process(
        self, screenshot: np.ndarray[Any, Any], text_prompt: str | None = None
    ) -> SemanticScene:
        """Process screenshot for semantic segmentation.

        Args:
            screenshot: Screenshot as numpy array
            text_prompt: Optional text description for concept-based segmentation

        Returns:
            SemanticScene with segmented objects
        """
        start_time = time.time()
        scene = SemanticScene(source_image=screenshot)

        if not HAS_SAM3 or self.processor is None:
            return scene

        # Convert numpy array to PIL Image
        from PIL import Image

        if screenshot.dtype == np.uint8:
            pil_image = Image.fromarray(screenshot)
        else:
            # Normalize to 0-255 if needed
            normalized = (
                (screenshot - screenshot.min()) * (255.0 / (screenshot.max() - screenshot.min()))
            ).astype(np.uint8)
            pil_image = Image.fromarray(normalized)

        # Set image in processor
        self.processor.set_image(pil_image)

        if text_prompt:
            # Use text-based concept segmentation
            self.processor.set_text_prompt(text_prompt)
            results = self.processor.segment()

            if results and "masks" in results:
                for i, mask in enumerate(results["masks"]):
                    if self._check_timeout(start_time):
                        break

                    bbox = results.get("boxes", [])[i] if "boxes" in results else None
                    confidence = results.get("scores", [])[i] if "scores" in results else 1.0

                    mask_data = {
                        "segmentation": mask,
                        "bbox": (bbox if bbox is not None else self._mask_to_bbox_list(mask)),
                        "predicted_iou": confidence,
                    }

                    semantic_obj = self._mask_to_semantic_object(mask_data, screenshot, index=i)

                    if semantic_obj and semantic_obj.confidence >= self._config.min_confidence:
                        scene.add_object(semantic_obj)

                    if len(scene.objects) >= self._config.max_objects:
                        break
        else:
            # Automatic segmentation using grid of points
            h, w = screenshot.shape[:2]
            grid_size = 32
            step_x = w // grid_size
            step_y = h // grid_size

            seen_masks = []

            for i in range(grid_size):
                for j in range(grid_size):
                    if self._check_timeout(start_time):
                        break

                    x = step_x * i + step_x // 2
                    y = step_y * j + step_y // 2

                    # Use point prompt for segmentation
                    results = self.processor.segment_from_point(x, y)

                    if results and "masks" in results and len(results["masks"]) > 0:
                        mask = results["masks"][0]

                        # Skip if we've seen a similar mask
                        if self._is_duplicate_mask(mask, seen_masks):
                            continue

                        seen_masks.append(mask)

                        bbox = self._mask_to_bbox_list(mask)
                        confidence = results.get("scores", [1.0])[0]

                        mask_data = {
                            "segmentation": mask,
                            "bbox": bbox,
                            "predicted_iou": confidence,
                        }

                        semantic_obj = self._mask_to_semantic_object(
                            mask_data, screenshot, index=len(scene.objects)
                        )

                        if semantic_obj and semantic_obj.confidence >= self._config.min_confidence:
                            scene.add_object(semantic_obj)

                        if len(scene.objects) >= self._config.max_objects:
                            break

                if len(scene.objects) >= self._config.max_objects:
                    break

        self._record_processing_time(start_time)
        return scene

    def _mask_to_semantic_object(
        self, mask_data: dict[str, Any], image: np.ndarray[Any, Any], index: int
    ) -> SemanticObject | None:
        """Convert SAM mask to SemanticObject.

        Args:
            mask_data: SAM mask data dictionary
            image: Original image
            index: Index of this mask

        Returns:
            SemanticObject or None if conversion fails
        """
        # Extract mask
        segmentation = mask_data["segmentation"]
        bbox = mask_data["bbox"]  # x, y, w, h format
        confidence = mask_data.get("predicted_iou", 1.0)

        # Create PixelLocation from mask
        location = PixelLocation.from_mask(
            segmentation, offset_x=int(bbox[0]), offset_y=int(bbox[1])
        )

        # Generate description
        description = self._generate_description(image, segmentation, bbox)

        # Create semantic object
        obj = SemanticObject(
            location=location,
            description=description,
            confidence=confidence,
            object_type=self._classify_segment(segmentation, bbox),
        )

        # Extract OCR text if enabled
        if self._config.enable_ocr:
            ocr_text = self._extract_text_from_region(image, bbox)
            if ocr_text:
                obj.ocr_text = ocr_text

        # Add additional attributes
        obj.add_attribute("area", mask_data.get("area", 0))
        obj.add_attribute("stability_score", mask_data.get("stability_score", 0))

        return obj

    def _generate_description(
        self, image: np.ndarray[Any, Any], mask: np.ndarray[Any, Any], bbox: list[int]
    ) -> str:
        """Generate semantic description for masked region.

        Args:
            image: Full image
            mask: Segmentation mask
            bbox: Bounding box [x, y, w, h]

        Returns:
            Text description
        """
        # Use the modular description generator
        return self.description_generator.generate(
            image, mask=mask, bbox=tuple(bbox) if bbox else None
        )

    def _classify_segment(self, mask: np.ndarray[Any, Any], bbox: list[int]) -> ObjectType:
        """Classify segment into object type.

        Args:
            mask: Segmentation mask
            bbox: Bounding box

        Returns:
            Estimated ObjectType
        """
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 1

        # Simple heuristics for classification
        if aspect_ratio > 3:
            # Wide objects might be toolbars or text fields
            if h < 50:
                return ObjectType.TOOLBAR
            else:
                return ObjectType.TEXT_FIELD
        elif 0.8 <= aspect_ratio <= 1.2 and w < 100:
            # Small square objects might be icons or checkboxes
            if w < 30:
                return ObjectType.CHECKBOX
            else:
                return ObjectType.ICON
        elif aspect_ratio > 1.5 and h < 60:
            # Wide, short objects might be buttons
            return ObjectType.BUTTON

        return ObjectType.UNKNOWN

    def _extract_text_from_region(self, image: np.ndarray[Any, Any], bbox: list[int]) -> str | None:
        """Extract OCR text from bounding box region.

        Args:
            image: Full image
            bbox: Bounding box [x, y, w, h]

        Returns:
            Extracted text or None
        """
        try:
            import pytesseract

            x, y, w, h = bbox
            region = image[y : y + h, x : x + w]

            # Convert to grayscale if needed
            if len(region.shape) == 3:
                import cv2

                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region

            # Extract text
            text = pytesseract.image_to_string(gray).strip()
            return text if text else None

        except ImportError:
            return None
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
            # Handle OCR and image processing errors:
            # - OSError: File/system errors during OCR processing
            # - RuntimeError: Tesseract processing failures
            # - ValueError: Invalid image dimensions or color space
            # - TypeError: Invalid array types or operations
            # - AttributeError: Missing cv2 or pytesseract methods
            return None

    def process_with_prompts(
        self,
        screenshot: np.ndarray[Any, Any],
        point_prompts: list[tuple[Any, ...]] | None = None,
        box_prompts: list[tuple[Any, ...]] | None = None,
    ) -> SemanticScene:
        """Process with specific point or box prompts.

        Args:
            screenshot: Screenshot to process
            point_prompts: List of (x, y) points to segment
            box_prompts: List of (x1, y1, x2, y2) boxes to segment

        Returns:
            SemanticScene with prompted segments
        """
        if not HAS_SAM3 or self.processor is None:
            return SemanticScene(source_image=screenshot)

        start_time = time.time()
        scene = SemanticScene(source_image=screenshot)

        # Set image
        self.predictor.set_image(screenshot)

        # Process point prompts
        if point_prompts:
            for point in point_prompts:
                if self._check_timeout(start_time):
                    break

                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([[point[0], point[1]]]),
                    point_labels=np.array([1]),  # 1 = foreground point
                    multimask_output=False,
                )

                if len(masks) > 0:
                    mask_data = {
                        "segmentation": masks[0],
                        "bbox": self._mask_to_bbox(masks[0]),
                        "predicted_iou": scores[0],
                    }
                    obj = self._mask_to_semantic_object(mask_data, screenshot, 0)
                    if obj:
                        scene.add_object(obj)

        # Process box prompts
        if box_prompts:
            for box in box_prompts:
                if self._check_timeout(start_time):
                    break

                masks, scores, _ = self.predictor.predict(box=np.array(box), multimask_output=False)

                if len(masks) > 0:
                    mask_data = {
                        "segmentation": masks[0],
                        "bbox": list[Any](box),
                        "predicted_iou": scores[0],
                    }
                    obj = self._mask_to_semantic_object(mask_data, screenshot, 0)
                    if obj:
                        scene.add_object(obj)

        self._record_processing_time(start_time)
        return scene

    def _mask_to_bbox_list(self, mask: np.ndarray[Any, Any]) -> list[int]:
        """Convert mask to bounding box.

        Args:
            mask: Binary mask

        Returns:
            [x, y, w, h] bounding box
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return [0, 0, 0, 0]

        rmin, rmax = row_indices[[0, -1]]
        cmin, cmax = col_indices[[0, -1]]
        return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]

    def _is_duplicate_mask(
        self,
        mask: np.ndarray[Any, Any],
        seen_masks: list[np.ndarray[Any, Any]],
        threshold: float = 0.8,
    ) -> bool:
        """Check if a mask is a duplicate of any seen masks.

        Args:
            mask: Current mask to check
            seen_masks: List of previously seen masks
            threshold: IoU threshold for considering masks as duplicates

        Returns:
            True if mask is a duplicate
        """
        for seen_mask in seen_masks:
            if mask.shape != seen_mask.shape:
                continue

            intersection = np.logical_and(mask, seen_mask).sum()
            union = np.logical_or(mask, seen_mask).sum()

            if union > 0:
                iou = intersection / union
                if iou > threshold:
                    return True

        return False

    def get_supported_object_types(self) -> set[str]:
        """Get supported object types.

        Returns:
            Set of all object types (SAM3 can segment anything)
        """
        return {obj_type.value for obj_type in ObjectType}

    def supports_incremental_processing(self) -> bool:
        """Check if incremental processing is supported.

        Returns:
            True - SAM3 can process specific regions incrementally
        """
        return True
