"""Segment Vectorizer for RAG element matching.

This module provides vectorization of SAM3 segments for vector database storage
and matching. It integrates SAM3 segmentation with CLIP embeddings to enable
efficient visual element matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np
from PIL import Image

from ..logging import get_logger
from ..semantic.description import BasicDescriptionGenerator, DescriptionGenerator
from .embeddings import CLIPEmbedder

logger = get_logger(__name__)


# Check for SAM3 availability
try:
    from ..semantic.processors.sam3_processor import HAS_SAM3, SAM3Processor
except ImportError:
    HAS_SAM3 = False
    SAM3Processor = None  # type: ignore


class MatchingStrategy(str, Enum):
    """Strategy for matching multi-pattern elements."""

    AVERAGE = "average"  # Average all pattern vectors into one query
    ANY_MATCH = "any_match"  # Match if ANY pattern exceeds threshold


@dataclass
class SegmentVector:
    """Vectorized representation of a screen segment.

    Attributes:
        mask: Binary segmentation mask
        bbox: Bounding box [x, y, width, height]
        image_embedding: CLIP image embedding (512-dim)
        text_embedding: CLIP text embedding (512-dim)
        text_description: AI-generated text description
        confidence: Segmentation confidence score
        ocr_text: Optional OCR text extracted from segment
    """

    mask: np.ndarray[Any, Any]
    bbox: tuple[int, int, int, int]  # x, y, width, height
    image_embedding: list[float]
    text_embedding: list[float] | None = None
    text_description: str = ""
    confidence: float = 1.0
    ocr_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of segment."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.bbox[2] * self.bbox[3]


@dataclass
class RAGMatch:
    """Result from RAG element matching.

    Attributes:
        segment: The matched segment
        visual_similarity: Image vector similarity score
        text_similarity: Text vector similarity (if computed)
        ocr_similarity: OCR text match ratio (if enabled)
        combined_score: Weighted combination of all scores
    """

    segment: SegmentVector
    visual_similarity: float
    text_similarity: float | None = None
    ocr_similarity: float | None = None
    ocr_text: str | None = None

    @property
    def combined_score(self) -> float:
        """Calculate combined score from available components.

        Uses weighted combination:
        - Visual: 70% (primary signal)
        - Text: 20% (semantic understanding)
        - OCR: 10% (text verification)
        """
        score = self.visual_similarity * 0.7

        if self.text_similarity is not None:
            score += self.text_similarity * 0.2
        else:
            # Redistribute text weight to visual if no text score
            score += self.visual_similarity * 0.2

        if self.ocr_similarity is not None:
            score += self.ocr_similarity * 0.1
        else:
            # Redistribute OCR weight to visual if no OCR score
            score += self.visual_similarity * 0.1

        return score

    @property
    def location(self) -> tuple[int, int, int, int]:
        """Get bounding box location."""
        return self.segment.bbox


class SegmentVectorizer:
    """Vectorize SAM3 segments for vector database storage and matching.

    This class handles the full pipeline of:
    1. Segmenting screenshots using SAM3
    2. Generating CLIP embeddings for each segment
    3. Generating text descriptions for semantic search
    4. Providing segment-to-segment matching

    Example:
        >>> vectorizer = SegmentVectorizer()
        >>> segments = vectorizer.vectorize_screenshot(screenshot)
        >>> for seg in segments:
        ...     print(f"Segment at {seg.bbox}: {seg.text_description}")
    """

    def __init__(
        self,
        sam3_checkpoint: str | None = None,
        clip_model: str = "openai/clip-vit-base-patch32",
        description_generator: DescriptionGenerator | None = None,
        enable_ocr: bool = False,
    ) -> None:
        """Initialize segment vectorizer.

        Args:
            sam3_checkpoint: Path to SAM3 checkpoint (uses default if None)
            clip_model: CLIP model identifier for embeddings
            description_generator: Optional custom description generator
            enable_ocr: Whether to enable OCR text extraction
        """
        self.sam3_checkpoint = sam3_checkpoint
        self.clip_model_name = clip_model
        self.enable_ocr = enable_ocr

        # Initialize CLIP embedder
        self._clip_embedder: CLIPEmbedder | None = None

        # Initialize SAM3 processor
        self._sam3_processor: Any = None

        # Initialize description generator
        self._description_generator = description_generator or BasicDescriptionGenerator()

        logger.info(
            "segment_vectorizer_initialized",
            sam3_available=HAS_SAM3,
            clip_model=clip_model,
            ocr_enabled=enable_ocr,
        )

    @property
    def clip_embedder(self) -> CLIPEmbedder:
        """Lazy-load CLIP embedder."""
        if self._clip_embedder is None:
            logger.info("loading_clip_embedder", model=self.clip_model_name)
            self._clip_embedder = CLIPEmbedder(model_name=self.clip_model_name)
        return self._clip_embedder

    @property
    def sam3_processor(self) -> Any:
        """Lazy-load SAM3 processor."""
        if self._sam3_processor is None and HAS_SAM3 and SAM3Processor is not None:
            logger.info("loading_sam3_processor", checkpoint=self.sam3_checkpoint)
            self._sam3_processor = SAM3Processor(
                checkpoint_path=self.sam3_checkpoint,
                description_generator=self._description_generator,
            )
        return self._sam3_processor

    def vectorize_segment(
        self,
        image: Image.Image | np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any],
        bbox: tuple[int, int, int, int] | None = None,
        confidence: float = 1.0,
    ) -> SegmentVector:
        """Generate embeddings for a single segment.

        Args:
            image: Source image (full screenshot or cropped region)
            mask: Binary mask for the segment
            bbox: Bounding box [x, y, width, height] (computed from mask if None)
            confidence: Segmentation confidence score

        Returns:
            SegmentVector with embeddings and metadata
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Convert mask to numpy if needed
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        # Compute bbox from mask if not provided
        if bbox is None:
            bbox = self._mask_to_bbox(mask)

        # Extract masked region
        masked_image = self._apply_mask(pil_image, mask, bbox)

        # Generate image embedding
        image_embedding = self.clip_embedder.encode_image(masked_image)

        # Generate text description
        np_image = np.array(pil_image)
        text_description = self._description_generator.generate(np_image, mask=mask, bbox=bbox)

        # Generate text embedding if we have a description
        text_embedding: list[float] | None = None
        if text_description and text_description != "unknown object":
            text_embedding = self.clip_embedder.encode_text(text_description)

        # Extract OCR text if enabled
        ocr_text: str | None = None
        if self.enable_ocr:
            ocr_text = self._extract_ocr(pil_image, bbox)

        return SegmentVector(
            mask=mask,
            bbox=bbox,
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            text_description=text_description,
            confidence=confidence,
            ocr_text=ocr_text,
        )

    def vectorize_screenshot(
        self,
        screenshot: Image.Image | np.ndarray[Any, Any],
        max_segments: int = 100,
        min_confidence: float = 0.5,
    ) -> list[SegmentVector]:
        """Segment and vectorize entire screenshot.

        Uses SAM3 automatic segmentation to find all regions,
        then vectorizes each segment.

        Args:
            screenshot: Screenshot to segment
            max_segments: Maximum number of segments to process
            min_confidence: Minimum confidence threshold for segments

        Returns:
            List of vectorized segments
        """
        # Convert to numpy for SAM3
        if isinstance(screenshot, Image.Image):
            np_image = np.array(screenshot)
            pil_image = screenshot
        else:
            np_image = screenshot
            pil_image = Image.fromarray(screenshot)

        segment_vectors: list[SegmentVector] = []

        # Use SAM3 if available
        if self.sam3_processor is not None:
            logger.info("segmenting_with_sam3")

            # Get semantic scene with segments
            scene = self.sam3_processor.process(np_image)

            for i, obj in enumerate(scene.objects[:max_segments]):
                if obj.confidence < min_confidence:
                    continue

                # Get mask from semantic object
                mask = obj.location.to_mask(np_image.shape[:2])

                # Get bbox from location
                x, y = obj.location.min_x, obj.location.min_y
                w = obj.location.max_x - obj.location.min_x
                h = obj.location.max_y - obj.location.min_y
                bbox = (x, y, w, h)

                try:
                    segment_vec = self.vectorize_segment(
                        pil_image,
                        mask,
                        bbox=bbox,
                        confidence=obj.confidence,
                    )

                    # Preserve description from SAM3 if available
                    if obj.description:
                        segment_vec.text_description = obj.description
                        # Re-encode text if description changed
                        segment_vec.text_embedding = self.clip_embedder.encode_text(obj.description)

                    # Add OCR text from semantic object if available
                    if hasattr(obj, "ocr_text") and obj.ocr_text:
                        segment_vec.ocr_text = obj.ocr_text

                    segment_vectors.append(segment_vec)

                except Exception as e:
                    logger.warning(
                        "segment_vectorization_failed",
                        segment_index=i,
                        error=str(e),
                    )
                    continue

        else:
            # Fallback: grid-based segmentation
            logger.info("segmenting_with_grid_fallback")
            segment_vectors = self._grid_segment(pil_image, np_image, max_segments)

        logger.info(
            "screenshot_vectorized",
            segment_count=len(segment_vectors),
        )

        return segment_vectors

    def vectorize_masked_image(
        self,
        image: Image.Image | np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any] | None = None,
    ) -> list[float]:
        """Generate embedding for an image with optional mask.

        This is used for indexing RAGElement images with their masks.

        Args:
            image: Source image
            mask: Optional mask to apply (None = use full image)

        Returns:
            CLIP embedding vector (512-dim)
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Apply mask if provided
        if mask is not None:
            bbox = self._mask_to_bbox(mask)
            pil_image = self._apply_mask(pil_image, mask, bbox)

        return self.clip_embedder.encode_image(pil_image)

    def aggregate_embeddings(
        self,
        embeddings: list[list[float]],
        strategy: MatchingStrategy = MatchingStrategy.AVERAGE,
    ) -> list[float]:
        """Aggregate multiple embeddings into one.

        Used for multi-pattern RAGElements to create a single query vector.

        Args:
            embeddings: List of embedding vectors
            strategy: Aggregation strategy (currently only AVERAGE supported)

        Returns:
            Aggregated embedding vector
        """
        if not embeddings:
            raise ValueError("Cannot aggregate empty embedding list")

        if len(embeddings) == 1:
            return embeddings[0]

        if strategy == MatchingStrategy.AVERAGE:
            # Convert to numpy for averaging
            emb_array = np.array(embeddings)
            averaged = np.mean(emb_array, axis=0)

            # Normalize to unit length
            norm = np.linalg.norm(averaged)
            if norm > 0:
                averaged = averaged / norm

            return cast(list[float], averaged.tolist())

        # For ANY_MATCH, we don't aggregate - caller handles individual comparisons
        return embeddings[0]

    def find_matches(
        self,
        query_embedding: list[float],
        segments: list[SegmentVector],
        threshold: float = 0.7,
        strategy: MatchingStrategy = MatchingStrategy.AVERAGE,
        ocr_filter: str | None = None,
    ) -> list[RAGMatch]:
        """Find segments matching a query embedding.

        Args:
            query_embedding: Query vector (from RAGElement)
            segments: List of screen segment vectors
            threshold: Minimum similarity threshold
            strategy: Matching strategy
            ocr_filter: Optional OCR text filter (must contain this text)

        Returns:
            List of matches sorted by combined score
        """
        matches: list[RAGMatch] = []

        for segment in segments:
            # Compute visual similarity
            visual_sim = self._cosine_similarity(query_embedding, segment.image_embedding)

            # Apply threshold
            if visual_sim < threshold:
                continue

            # Apply OCR filter if specified
            if ocr_filter:
                if not segment.ocr_text:
                    continue
                if ocr_filter.lower() not in segment.ocr_text.lower():
                    continue

            # Compute text similarity if available
            text_sim: float | None = None
            if segment.text_embedding:
                text_sim = self._cosine_similarity(query_embedding, segment.text_embedding)

            # Compute OCR similarity if available
            ocr_sim: float | None = None
            if ocr_filter and segment.ocr_text:
                ocr_sim = self._text_similarity(ocr_filter, segment.ocr_text)

            match = RAGMatch(
                segment=segment,
                visual_similarity=visual_sim,
                text_similarity=text_sim,
                ocr_similarity=ocr_sim,
                ocr_text=segment.ocr_text,
            )

            matches.append(match)

        # Sort by combined score
        matches.sort(key=lambda m: m.combined_score, reverse=True)

        return matches

    def find_any_match(
        self,
        pattern_embeddings: list[list[float]],
        segments: list[SegmentVector],
        threshold: float = 0.7,
    ) -> RAGMatch | None:
        """Find first segment matching ANY of the pattern embeddings.

        Used for MatchingStrategy.ANY_MATCH where we want to match
        if any single pattern exceeds the threshold.

        Args:
            pattern_embeddings: List of pattern embedding vectors
            segments: List of screen segment vectors
            threshold: Minimum similarity threshold

        Returns:
            Best match, or None if no match found
        """
        best_match: RAGMatch | None = None
        best_score = 0.0

        for segment in segments:
            for pattern_emb in pattern_embeddings:
                visual_sim = self._cosine_similarity(pattern_emb, segment.image_embedding)

                if visual_sim >= threshold and visual_sim > best_score:
                    best_match = RAGMatch(
                        segment=segment,
                        visual_similarity=visual_sim,
                    )
                    best_score = visual_sim

        return best_match

    def _apply_mask(
        self,
        image: Image.Image,
        mask: np.ndarray[Any, Any],
        bbox: tuple[int, int, int, int],
    ) -> Image.Image:
        """Apply mask to image and crop to bounding box.

        Args:
            image: Source image
            mask: Binary mask
            bbox: Bounding box [x, y, width, height]

        Returns:
            Masked and cropped image with transparency
        """
        x, y, w, h = bbox

        # Ensure bounds are within image
        img_w, img_h = image.size
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Crop image to bounding box
        cropped = image.crop((x, y, x + w, y + h))

        # Crop mask to match
        mask_cropped = mask[y : y + h, x : x + w]

        # Convert to RGBA to add alpha channel
        rgba = cropped.convert("RGBA")

        # Apply mask as alpha channel
        np_rgba = np.array(rgba)
        if mask_cropped.shape[:2] == np_rgba.shape[:2]:
            # Normalize mask to 0-255
            alpha = (mask_cropped * 255).astype(np.uint8)
            np_rgba[:, :, 3] = alpha

        return Image.fromarray(np_rgba)

    def _mask_to_bbox(self, mask: np.ndarray[Any, Any]) -> tuple[int, int, int, int]:
        """Convert mask to bounding box.

        Args:
            mask: Binary mask

        Returns:
            Tuple of (x, y, width, height)
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return (0, 0, mask.shape[1], mask.shape[0])

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        y_min, y_max = int(row_indices[0]), int(row_indices[-1])
        x_min, x_max = int(col_indices[0]), int(col_indices[-1])

        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    def _grid_segment(
        self,
        pil_image: Image.Image,
        np_image: np.ndarray[Any, Any],
        max_segments: int,
    ) -> list[SegmentVector]:
        """Fallback contour-based segmentation when SAM3 is unavailable.

        Uses OpenCV edge detection and contour finding to identify
        meaningful UI regions instead of a simple grid.

        Args:
            pil_image: PIL image
            np_image: Numpy image
            max_segments: Maximum segments to generate

        Returns:
            List of segment vectors
        """
        import cv2

        segments: list[SegmentVector] = []
        h, w = np_image.shape[:2]

        # Convert to grayscale for edge detection
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_image

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Morphological operations to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours (RETR_TREE to get all contours including nested)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Minimum size threshold (filter tiny regions)
        min_area = (h * w) * 0.001  # At least 0.1% of image area

        for i, contour in enumerate(contours):
            if len(segments) >= max_segments:
                break

            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(contour)

            # Filter out very small regions
            area = cv2.contourArea(contour)
            if area < min_area or cw < 10 or ch < 10:
                continue

            # Create mask for this contour
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (1.0,), thickness=cv2.FILLED)

            bbox = (x, y, cw, ch)

            try:
                segment_vec = self.vectorize_segment(
                    pil_image,
                    mask,
                    bbox=bbox,
                    confidence=0.8,  # Lower confidence for contour detection
                )
                segment_vec.metadata["contour_index"] = i
                segment_vec.metadata["contour_area"] = float(area)
                segments.append(segment_vec)
            except Exception as e:
                logger.warning(
                    "contour_segment_failed",
                    contour_index=i,
                    error=str(e),
                )

        # If no contours found, fall back to simple grid
        if len(segments) == 0:
            logger.warning("no_contours_found_falling_back_to_grid")
            return self._simple_grid_segment(pil_image, np_image, max_segments)

        return segments

    def _simple_grid_segment(
        self,
        pil_image: Image.Image,
        np_image: np.ndarray[Any, Any],
        max_segments: int,
    ) -> list[SegmentVector]:
        """Simple grid-based segmentation as last resort fallback.

        Args:
            pil_image: PIL image
            np_image: Numpy image
            max_segments: Maximum segments to generate

        Returns:
            List of segment vectors
        """
        segments: list[SegmentVector] = []
        h, w = np_image.shape[:2]

        # Use smaller grid (4x4 = 16 segments max)
        grid_size = min(4, int(np.sqrt(max_segments)))
        cell_w = w // grid_size
        cell_h = h // grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                if len(segments) >= max_segments:
                    break

                x = col * cell_w
                y = row * cell_h

                # Create mask just for this cell region (not full image)
                # This avoids dimension mismatch issues
                bbox = (x, y, cell_w, cell_h)

                # Create a cell-sized mask
                cell_mask = np.ones((cell_h, cell_w), dtype=np.uint8)

                try:
                    # Crop the image to the cell
                    cell_image = pil_image.crop((x, y, x + cell_w, y + cell_h))

                    segment_vec = self.vectorize_segment(
                        cell_image,
                        cell_mask,
                        bbox=None,  # bbox is relative to cropped image
                        confidence=0.5,
                    )
                    # Store actual bbox in metadata
                    segment_vec.bbox = bbox
                    segment_vec.metadata["grid_row"] = row
                    segment_vec.metadata["grid_col"] = col
                    segments.append(segment_vec)
                except Exception as e:
                    logger.warning(
                        "grid_segment_failed",
                        row=row,
                        col=col,
                        error=str(e),
                    )

        return segments

    def _extract_ocr(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> str | None:
        """Extract OCR text from image region.

        Args:
            image: Source image
            bbox: Bounding box [x, y, width, height]

        Returns:
            Extracted text or None
        """
        try:
            import pytesseract

            x, y, w, h = bbox
            region = image.crop((x, y, x + w, y + h))

            # Convert to grayscale for better OCR
            gray = region.convert("L")

            text = pytesseract.image_to_string(gray).strip()
            return text if text else None

        except ImportError:
            logger.debug("pytesseract_not_available")
            return None
        except Exception as e:
            logger.debug("ocr_extraction_failed", error=str(e))
            return None

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0-1.0)
        """
        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity using Levenshtein ratio.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio (0.0-1.0)
        """
        try:
            from difflib import SequenceMatcher

            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        except Exception:
            return 0.0
