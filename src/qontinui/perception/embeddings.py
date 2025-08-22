"""Embedding generation using CLIP and other vision-language models.

This module provides multimodal embeddings for visual and text inputs,
enabling semantic understanding and natural language queries.
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path
import hashlib
from dataclasses import dataclass

from ..logging import get_logger
from ..config import get_settings
from ..exceptions import ModelLoadException, InferenceException

logger = get_logger(__name__)


@dataclass 
class EmbeddingResult:
    """Result of embedding generation.
    
    Attributes:
        embedding: The embedding vector
        modality: Input modality (image/text/multimodal)
        metadata: Additional metadata
    """
    embedding: np.ndarray
    modality: str
    metadata: Dict[str, Any]


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
        device: Optional[str] = None,
        use_cache: bool = True,
        cache_size: int = 1000
    ):
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
            device = "cuda" if torch.cuda.is_available() and self.settings.use_gpu else "cpu"
        self.device = device
        
        # Initialize cache
        self._cache = {} if use_cache else None
        self.cache_size = cache_size
        
        # Initialize model
        self.model = None
        self.processor = None
        self._initialize_model()
        
        logger.info(
            "embedding_generator_initialized",
            model=model_name,
            device=device,
            cache_enabled=use_cache
        )
    
    def _initialize_model(self):
        """Initialize CLIP model and processor."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
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
                embedding_dim=self.embedding_dim
            )
            
        except Exception as e:
            raise ModelLoadException(
                model_name=self.model_name,
                reason=str(e)
            )
    
    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path, List],
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
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
            images = image
            
        # Convert images to PIL format
        pil_images = []
        cache_keys = []
        
        for img in images:
            # Convert to PIL
            if isinstance(img, (str, Path)):
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
        embeddings = []
        uncached_images = []
        uncached_indices = []
        
        for i, (img, key) in enumerate(zip(pil_images, cache_keys)):
            if self.use_cache and key in self._cache:
                embeddings.append(self._cache[key])
                logger.debug("embedding_cache_hit", key=key[:8])
            else:
                uncached_images.append(img)
                uncached_indices.append(i)
                embeddings.append(None)
        
        # Process uncached images
        if uncached_images:
            try:
                # Prepare inputs
                inputs = self.processor(
                    images=uncached_images,
                    return_tensors="pt",
                    padding=True
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
                for idx, emb, img in zip(uncached_indices, new_embeddings, uncached_images):
                    embeddings[idx] = emb
                    
                    if self.use_cache:
                        key = cache_keys[idx]
                        self._update_cache(key, emb)
                        
            except Exception as e:
                raise InferenceException(
                    model_name=self.model_name,
                    reason=f"Image encoding failed: {e}"
                )
        
        # Return single or batch
        if is_batch:
            return embeddings
        else:
            return embeddings[0]
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
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
            texts = [text]
        else:
            texts = text
            
        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, txt in enumerate(texts):
            if self.use_cache:
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
            try:
                # Prepare inputs
                inputs = self.processor(
                    text=uncached_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77  # CLIP max length
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
                for idx, emb, txt in zip(uncached_indices, new_embeddings, uncached_texts):
                    embeddings[idx] = emb
                    
                    if self.use_cache:
                        key = self._get_text_cache_key(txt)
                        self._update_cache(key, emb)
                        
            except Exception as e:
                raise InferenceException(
                    model_name=self.model_name,
                    reason=f"Text encoding failed: {e}"
                )
        
        # Return single or batch
        if is_batch:
            return embeddings
        else:
            return embeddings[0]
    
    def encode_multimodal(
        self,
        image: Union[Image.Image, np.ndarray],
        text: str,
        fusion: str = "average"
    ) -> np.ndarray:
        """Generate combined image-text embedding.
        
        Args:
            image: Input image
            text: Input text
            fusion: Fusion method (average, concat, weighted)
            
        Returns:
            Fused embedding
        """
        # Get individual embeddings
        image_emb = self.encode_image(image, normalize=True)
        text_emb = self.encode_text(text, normalize=True)
        
        # Fuse embeddings
        if fusion == "average":
            fused = (image_emb + text_emb) / 2
            fused = fused / np.linalg.norm(fused)  # Renormalize
        elif fusion == "concat":
            fused = np.concatenate([image_emb, text_emb])
        elif fusion == "weighted":
            # Use settings for weights
            img_weight = self.settings.deterministic_weight
            txt_weight = self.settings.semantic_weight
            fused = img_weight * image_emb + txt_weight * text_emb
            fused = fused / np.linalg.norm(fused)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")
            
        return fused
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
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
                embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :],
                axis=2
            )
            
        elif metric == "dot":
            # Simple dot product
            similarity = np.dot(embeddings1, embeddings2.T)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        return similarity
    
    def find_similar(
        self,
        query: Union[str, Image.Image, np.ndarray],
        candidates: List[np.ndarray],
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float]]:
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
            query_emb = self.encode_text(query)
        else:
            query_emb = self.encode_image(query)
            
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
        results = [(int(idx), float(similarities[i])) 
                   for i, idx in enumerate(top_k_indices)]
        
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
    
    def _update_cache(self, key: str, embedding: np.ndarray):
        """Update embedding cache."""
        # Limit cache size
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            
        self._cache[key] = embedding
    
    def clear_cache(self):
        """Clear embedding cache."""
        if self._cache:
            self._cache.clear()
            logger.debug("embedding_cache_cleared")
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim