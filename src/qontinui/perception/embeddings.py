"""Embedding generation using CLIP and other vision-language models.

This module provides multimodal embeddings for visual and text inputs,
enabling semantic understanding and natural language queries.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from ..config import get_settings
from ..exceptions import InferenceException, ModelLoadException
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation.

    Attributes:
        embedding: The embedding vector
        modality: Input modality (image/text/multimodal)
        metadata: Additional metadata
    """

    embedding: np.ndarray[Any, Any]
    modality: str
    metadata: dict[str, Any]


class EmbeddingGenerator:
    """Generate embeddings for visual and text inputs using CLIP.

    Features:
        - Image embeddings for UI elements
        - Text embeddings for natural language queries
        - Cross-modal similarity computation
        - Batch processing support
        - Caching for performance
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        use_cache: bool = True,
        cache_size: int = 1000,
    ) -> None:
        """Initialize embedding generator.

        Args:
            model_name: CLIP model to use
            device: Device (cuda/cpu/auto)
            use_cache: Enable embedding cache
            cache_size: Maximum cache entries
        """
        self.settings = get_settings()
        self.model_name = model_name
        self.use_cache = use_cache

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize cache
        self._cache: dict[str, np.ndarray[Any, Any]] | None = {} if use_cache else None
        self.cache_size = cache_size

        # Initialize model
        self.model = None
        self.processor = None
        self._initialize_model()

        logger.info(
            "embedding_generator_initialized",
            model=model_name,
            device=device,
            cache_enabled=use_cache,
        )

    def _initialize_model(self):
        """Initialize CLIP model and processor."""
        try:
            from transformers import CLIPModel, CLIPProcessor

            # Load model and processor
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension
            self.embedding_dim = self.model.config.projection_dim

            logger.debug(
                "clip_model_loaded",
                model=self.model_name,
                device=self.device,
                embedding_dim=self.embedding_dim,
            )

        except Exception as e:
            raise ModelLoadException(model_name=self.model_name, reason=str(e)) from e

    def encode_image(
        self,
        image: Image.Image | np.ndarray[Any, Any] | str | Path | list[Any],
        normalize: bool = True,
    ) -> np.ndarray[Any, Any] | list[np.ndarray[Any, Any]]:
        """Generate image embeddings.

        Args:
            image: Single image or list of images
            normalize: Normalize embeddings to unit length

        Returns:
            Embedding vector(s)
        """
        # Handle batch input
        is_batch = isinstance(image, list)
        if not is_batch:
            images = [image]
        else:
            images = cast(list[Any], image)

        # Convert images to PIL format
        pil_images = []
        cache_keys = []

        for img in images:
            # Convert to PIL
            if isinstance(img, str | Path):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            pil_images.append(pil_img)

            # Generate cache key
            if self.use_cache:
                cache_key = self._get_image_cache_key(pil_img)
                cache_keys.append(cache_key)

        # Check cache and process uncached images
        embeddings: list[np.ndarray[Any, Any] | None] = []
        uncached_images = []
        uncached_indices = []

        for i, (img, key) in enumerate(zip(pil_images, cache_keys, strict=False)):
            if self.use_cache and self._cache is not None and key in self._cache:
                embeddings.append(self._cache[key])
                logger.debug("embedding_cache_hit", key=key[:8])
            else:
                uncached_images.append(img)
                uncached_indices.append(i)
                embeddings.append(None)

        # Process uncached images
        if uncached_images:
            if self.processor is None or self.model is None:
                raise RuntimeError("Model not initialized")

            try:
                # Prepare inputs
                inputs = self.processor(
                    images=uncached_images, return_tensors="pt", padding=True
                ).to(self.device)

                # Generate embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                # Normalize if requested
                if normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Convert to numpy
                new_embeddings = image_features.cpu().numpy()

                # Update results and cache
                for idx, emb, _img in zip(
                    uncached_indices, new_embeddings, uncached_images, strict=False
                ):
                    embeddings[idx] = emb

                    if self.use_cache:
                        key = cache_keys[idx]
                        self._update_cache(key, emb)

            except Exception as e:
                raise InferenceException(
                    model_name=self.model_name, reason=f"Image encoding failed: {e}"
                ) from e

        # Return single or batch (filter out None values)
        if is_batch:
            # Filter out None values to match return type
            return [emb for emb in embeddings if emb is not None]
        else:
            # For single image, return the first embedding (guaranteed to be non-None at this point)
            result = embeddings[0]
            if result is None:
                raise InferenceException(self.model_name, "Failed to generate image embedding")
            return result

    def encode_text(
        self, text: str | list[str], normalize: bool = True
    ) -> np.ndarray[Any, Any] | list[np.ndarray[Any, Any]]:
        """Generate text embeddings.

        Args:
            text: Single text or list of texts
            normalize: Normalize embeddings to unit length

        Returns:
            Embedding vector(s)
        """
        # Handle batch input
        is_batch = isinstance(text, list)
        if not is_batch:
            texts: list[str] = [text]  # type: ignore[list-item]
        else:
            texts = cast(list[str], text)

        # Check cache
        embeddings: list[np.ndarray[Any, Any] | None] = []
        uncached_texts = []
        uncached_indices = []

        for i, txt in enumerate(texts):
            # txt is always str here since texts is always a list[str] at this point
            if self.use_cache and self._cache is not None:
                cache_key = self._get_text_cache_key(txt)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                    logger.debug("embedding_cache_hit", key=cache_key[:8])
                    continue

            uncached_texts.append(txt)
            uncached_indices.append(i)
            embeddings.append(None)

        # Process uncached texts
        if uncached_texts:
            if self.processor is None or self.model is None:
                raise RuntimeError("Model not initialized")

            try:
                # Prepare inputs
                inputs = self.processor(
                    text=uncached_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,  # CLIP max length
                ).to(self.device)

                # Generate embeddings
                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)

                # Normalize if requested
                if normalize:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Convert to numpy
                new_embeddings = text_features.cpu().numpy()

                # Update results and cache
                for idx, emb, txt in zip(
                    uncached_indices, new_embeddings, uncached_texts, strict=False
                ):
                    embeddings[idx] = emb

                    if self.use_cache:
                        key = self._get_text_cache_key(txt)
                        self._update_cache(key, emb)

            except Exception as e:
                raise InferenceException(
                    model_name=self.model_name, reason=f"Text encoding failed: {e}"
                ) from e

        # Return single or batch (filter out None values)
        if is_batch:
            # Filter out None values to match return type
            return [emb for emb in embeddings if emb is not None]
        else:
            # For single text, return the first embedding (guaranteed to be non-None at this point)
            result = embeddings[0]
            if result is None:
                raise InferenceException(self.model_name, "Failed to generate text embedding")
            return result

    def encode_multimodal(
        self, image: Image.Image | np.ndarray[Any, Any], text: str, fusion: str = "average"
    ) -> np.ndarray[Any, Any]:
        """Generate combined image-text embedding.

        Args:
            image: Input image
            text: Input text
            fusion: Fusion method (average, concat, weighted)

        Returns:
            Fused embedding
        """
        # Get individual embeddings
        image_emb_result = self.encode_image(image, normalize=True)
        text_emb_result = self.encode_text(text, normalize=True)

        # Ensure we have single embeddings, not lists
        image_emb = image_emb_result[0] if isinstance(image_emb_result, list) else image_emb_result
        text_emb = text_emb_result[0] if isinstance(text_emb_result, list) else text_emb_result

        # Fuse embeddings
        if fusion == "average":
            fused = (image_emb + text_emb) / 2
            fused = fused / np.linalg.norm(fused)  # Renormalize
        elif fusion == "concat":
            fused = np.concatenate([image_emb, text_emb])
        elif fusion == "weighted":
            # Use default weights (deterministic vs semantic)
            img_weight = 0.7  # Default deterministic weight
            txt_weight = 0.3  # Default semantic weight
            fused = img_weight * image_emb + txt_weight * text_emb
            fused = fused / np.linalg.norm(fused)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

        return cast(np.ndarray[Any, Any], fused)

    def compute_similarity(
        self,
        embeddings1: np.ndarray[Any, Any],
        embeddings2: np.ndarray[Any, Any],
        metric: str = "cosine",
    ) -> np.ndarray[Any, Any]:
        """Compute similarity between embeddings.

        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)
            metric: Similarity metric (cosine, euclidean, dot)

        Returns:
            Similarity matrix (N x M)
        """
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        if metric == "cosine":
            # Normalize embeddings
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            # Compute cosine similarity
            similarity = np.dot(embeddings1, embeddings2.T)

        elif metric == "euclidean":
            # Compute negative euclidean distance
            # Use broadcasting for efficiency
            similarity = -np.linalg.norm(
                embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :], axis=2
            )

        elif metric == "dot":
            # Simple dot product
            similarity = np.dot(embeddings1, embeddings2.T)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return cast(np.ndarray[Any, Any], similarity)

    def find_similar(
        self,
        query: str | Image.Image | np.ndarray[Any, Any],
        candidates: list[np.ndarray[Any, Any]],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[int, float]]:
        """Find most similar candidates to query.

        Args:
            query: Query text or image
            candidates: List of candidate embeddings
            k: Number of top results
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity) tuples
        """
        # Get query embedding
        if isinstance(query, str):
            query_emb_result = self.encode_text(query)
        else:
            query_emb_result = self.encode_image(query)

        # Ensure single embedding (not list)
        if isinstance(query_emb_result, list):
            query_emb: np.ndarray[Any, Any] = query_emb_result[0]
        else:
            query_emb = query_emb_result

        # Stack candidates
        candidate_matrix = np.vstack(candidates)

        # Compute similarities
        similarities = self.compute_similarity(query_emb, candidate_matrix)[0]

        # Apply threshold
        if threshold is not None:
            valid_indices = np.where(similarities >= threshold)[0]
            similarities = similarities[valid_indices]
            indices = valid_indices
        else:
            indices = np.arange(len(similarities))

        # Get top k
        if len(indices) > k:
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        else:
            top_k_indices = indices[np.argsort(similarities)[::-1]]

        # Return results
        results = [(int(idx), float(similarities[i])) for i, idx in enumerate(top_k_indices)]

        return results

    def _get_image_cache_key(self, image: Image.Image) -> str:
        """Generate cache key for image."""
        # Hash image bytes
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        return f"img_{img_hash}"

    def _get_text_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"txt_{text_hash}"

    def _update_cache(self, key: str, embedding: np.ndarray[Any, Any]):
        """Update embedding cache."""
        if self._cache is None:
            return

        # Limit cache size
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = cast(str, next(iter(self._cache)))
            del self._cache[oldest_key]

        self._cache[key] = embedding

    def clear_cache(self):
        """Clear embedding cache."""
        if self._cache:
            self._cache.clear()
            logger.debug("embedding_cache_cleared")

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return cast(int, self.embedding_dim)
