"""Text embedding generation for Qontinui RAG system.

This module provides text embeddings using sentence-transformers,
enabling semantic search over GUI elements and their descriptions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

try:
    import torch
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ...logging import get_logger
from ..models import GUIElementChunk

logger = get_logger(__name__)


def colors_to_semantic(hex_colors: list[str]) -> list[str]:
    """Convert hex color codes to semantic color names.

    Args:
        hex_colors: List of hex color codes (e.g., ['#FF0000', '#00FF00'])

    Returns:
        List of semantic color names (e.g., ['red', 'green'])
    """
    semantic_names = []

    for hex_color in hex_colors:
        # Remove '#' if present
        hex_color = hex_color.lstrip("#")

        # Convert to RGB
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        except (ValueError, IndexError):
            logger.warning("invalid_hex_color", hex=hex_color)
            continue

        # Determine semantic name based on RGB values
        # Use simple heuristics for common colors
        max_val = max(r, g, b)
        min_val = min(r, g, b)

        # Check for grayscale
        if max_val - min_val < 30:
            if max_val < 60:
                semantic_names.append("black")
            elif max_val < 150:
                semantic_names.append("gray")
            elif max_val < 220:
                semantic_names.append("light_gray")
            else:
                semantic_names.append("white")
            continue

        # Check for primary/secondary colors
        if r > g and r > b:
            if g > 100 and b < 100:
                semantic_names.append("orange")
            elif b > 100:
                semantic_names.append("magenta")
            else:
                semantic_names.append("red")
        elif g > r and g > b:
            if r > 100:
                semantic_names.append("yellow")
            elif b > 100:
                semantic_names.append("cyan")
            else:
                semantic_names.append("green")
        elif b > r and b > g:
            if r > 100:
                semantic_names.append("purple")
            elif g > 100:
                semantic_names.append("cyan")
            else:
                semantic_names.append("blue")
        else:
            semantic_names.append("gray")

    return semantic_names


class TextDescriptionGenerator:
    """Generate searchable text descriptions from GUI elements.

    This class converts GUI element metadata into natural language descriptions
    that can be embedded and searched semantically.
    """

    def generate(self, element: GUIElementChunk) -> str:
        """Generate searchable text description from element.

        Args:
            element: GUI element to describe

        Returns:
            Natural language description suitable for embedding
        """
        parts = []

        # Add element type
        if element.element_type:
            parts.append(element.element_type.value)

        # Add element subtype if available
        if element.element_subtype:
            parts.append(element.element_subtype)

        # Add visual state if not normal
        if element.visual_state and element.visual_state.lower() != "normal":
            parts.append(element.visual_state.lower())

        # Add color names if available
        if element.dominant_colors:
            # Convert first dominant color to hex and get semantic name
            color = element.dominant_colors[0]
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            semantic_colors = colors_to_semantic([hex_color])
            if semantic_colors:
                parts.extend(semantic_colors)

        # Add OCR text
        if element.ocr_text:
            # Clean and normalize OCR text
            ocr_text = element.ocr_text.strip()
            if ocr_text:
                parts.append(f"with text '{ocr_text}'")

        # Add parent region if available
        if element.parent_region:
            parts.append(f"in {element.parent_region}")

        # Add semantic action if available
        if element.semantic_action:
            parts.append(f"for {element.semantic_action}")

        # Add semantic role if available and different from action
        if element.semantic_role and element.semantic_role != element.semantic_action:
            parts.append(f"role {element.semantic_role}")

        # Add interaction type if available
        if element.interaction_type:
            parts.append(f"interaction {element.interaction_type}")

        # Add state indicators
        state_indicators = []
        if not element.is_enabled:
            state_indicators.append("disabled")
        if element.is_selected:
            state_indicators.append("selected")
        if element.is_focused:
            state_indicators.append("focused")

        if state_indicators:
            parts.append(" ".join(state_indicators))

        # Add platform if available
        if element.platform:
            parts.append(f"platform {element.platform}")

        # Add style family if available
        if element.style_family:
            parts.append(f"style {element.style_family}")

        # Combine into natural language
        description = " ".join(parts)

        # Clean up extra whitespace
        description = re.sub(r"\s+", " ", description).strip()

        return description


class TextEmbedder:
    """Text embedding generator using sentence-transformers.

    This class provides efficient text embedding generation for semantic search
    over GUI element descriptions.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize text embedder.

        Args:
            model_name: Sentence-transformers model name
            cache_dir: Directory to cache downloaded models
            device: Device to use (cpu/cuda/auto). If None, auto-detect.

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: poetry install -E rag"
            )

        self.model_name = model_name
        self.cache_dir = cache_dir

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model
        try:
            logger.info(
                "loading_text_embedder",
                model=model_name,
                device=device,
                cache_dir=str(cache_dir) if cache_dir else "default",
            )

            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(cache_dir) if cache_dir else None,
                device=device,
            )

            logger.info(
                "text_embedder_loaded",
                model=model_name,
                embedding_dim=self.model.get_sentence_embedding_dimension(),
            )

        except Exception as e:
            logger.error("text_embedder_load_failed", model=model_name, error=str(e))
            raise RuntimeError(
                f"Failed to load text embedder model '{model_name}': {e}"
            ) from e

    def encode(self, text: str) -> list[float]:
        """Encode single text into embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Encode text
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Convert to list
            return cast(list[float], embedding.tolist())

        except Exception as e:
            logger.error("text_encoding_failed", text=text[:100], error=str(e))
            raise RuntimeError(f"Failed to encode text: {e}") from e

    def batch_encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Encode multiple texts in batches.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        try:
            logger.debug(
                "batch_encoding",
                count=len(texts),
                batch_size=batch_size,
            )

            # Encode all texts in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )

            # Convert to list of lists
            return cast(list[list[float]], embeddings.tolist())

        except Exception as e:
            logger.error("batch_encoding_failed", count=len(texts), error=str(e))
            raise RuntimeError(f"Failed to encode texts: {e}") from e

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        return cast(int, self.model.get_sentence_embedding_dimension())

    @property
    def model_version(self) -> str:
        """Return model identifier for version tracking.

        Returns:
            Model identifier string
        """
        return self.model_name
