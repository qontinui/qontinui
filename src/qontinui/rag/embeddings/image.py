"""Image embedding models for RAG."""

from pathlib import Path
from typing import cast

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    """CLIP-based image and text embedder for multimodal search."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize CLIP model.

        Args:
            model_name: HuggingFace model identifier
            cache_dir: Optional cache directory for model weights
        """
        self._model_name = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._processor = CLIPProcessor.from_pretrained(model_name, cache_dir=self._cache_dir)
            self._model = CLIPModel.from_pretrained(model_name, cache_dir=self._cache_dir).to(
                self._device
            )
            self._model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLIP model '{model_name}'. "
                f"Ensure transformers is installed and model exists. Error: {e}"
            ) from e

    def encode_image(self, image: Image.Image) -> list[float]:
        """Encode a single image to embedding vector.

        Args:
            image: PIL Image to encode

        Returns:
            Embedding vector as list of floats
        """
        try:
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)

            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            result: list[float] = image_features.cpu().numpy().flatten().tolist()
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to encode image with CLIP: {e}") from e

    def encode_text(self, text: str) -> list[float]:
        """Encode text for text-to-image search.

        Args:
            text: Text query to encode

        Returns:
            Embedding vector as list of floats
        """
        try:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(
                self._device
            )

            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            result: list[float] = text_features.cpu().numpy().flatten().tolist()
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to encode text with CLIP: {e}") from e

    def batch_encode_images(
        self, images: list[Image.Image], batch_size: int = 16
    ) -> list[list[float]]:
        """Batch encode multiple images.

        Args:
            images: List of PIL Images to encode
            batch_size: Number of images to process at once

        Returns:
            List of embedding vectors
        """
        embeddings: list[list[float]] = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            try:
                inputs = self._processor(images=batch, return_tensors="pt").to(self._device)

                with torch.no_grad():
                    batch_features = self._model.get_image_features(**inputs)
                    # Normalize for cosine similarity
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

                embeddings.extend(batch_features.cpu().numpy().tolist())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to batch encode images (batch {i // batch_size}): {e}"
                ) from e

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (512 for ViT-B/32)."""
        return 512

    @property
    def model_version(self) -> str:
        """Return model identifier."""
        return self._model_name


class DINOv2Embedder:
    """DINOv2-based image embedder for visual similarity search."""

    _MODEL_CONFIGS = {
        "dinov2_vits14": {"embed_dim": 384, "repo": "facebookresearch/dinov2"},
        "dinov2_vitb14": {"embed_dim": 768, "repo": "facebookresearch/dinov2"},
        "dinov2_vitl14": {"embed_dim": 1024, "repo": "facebookresearch/dinov2"},
        "dinov2_vitg14": {"embed_dim": 1536, "repo": "facebookresearch/dinov2"},
    }

    def __init__(self, model_name: str = "dinov2_vits14", cache_dir: Path | None = None) -> None:
        """Initialize DINOv2 model.

        Args:
            model_name: Model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            cache_dir: Optional cache directory for model weights
        """
        if model_name not in self._MODEL_CONFIGS:
            raise ValueError(
                f"Unknown DINOv2 model: {model_name}. "
                f"Choose from: {list(self._MODEL_CONFIGS.keys())}"
            )

        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._config = self._MODEL_CONFIGS[model_name]

        try:
            # Load DINOv2 from torch hub
            self._model = torch.hub.load(
                self._config["repo"],
                model_name,
                pretrained=True,
                verbose=False,
            ).to(self._device)
            self._model.eval()

            # DINOv2 preprocessing
            self._transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{model_name}'. "
                f"Ensure torch and torchvision are installed. Error: {e}"
            ) from e

    def encode(self, image: Image.Image) -> list[float]:
        """Encode a single image to embedding vector.

        Args:
            image: PIL Image to encode

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Preprocess and add batch dimension
            img_tensor = self._transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # Get CLS token embedding
                features = self._model(img_tensor)

            result: list[float] = features.cpu().numpy().flatten().tolist()
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to encode image with DINOv2: {e}") from e

    def batch_encode(self, images: list[Image.Image], batch_size: int = 16) -> list[list[float]]:
        """Batch encode multiple images.

        Args:
            images: List of PIL Images to encode
            batch_size: Number of images to process at once

        Returns:
            List of embedding vectors
        """
        embeddings: list[list[float]] = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            try:
                # Preprocess all images in batch
                batch_tensors = []
                for img in batch:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    batch_tensors.append(self._transform(img))

                # Stack into batch tensor
                batch_tensor = torch.stack(batch_tensors).to(self._device)

                with torch.no_grad():
                    batch_features = self._model(batch_tensor)

                embeddings.extend(batch_features.cpu().numpy().tolist())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to batch encode images (batch {i // batch_size}): {e}"
                ) from e

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return cast(int, self._config["embed_dim"])

    @property
    def model_version(self) -> str:
        """Return model identifier."""
        return self._model_name


class HybridImageEmbedder:
    """Hybrid embedder using both CLIP and DINOv2 for comprehensive image representation."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize both CLIP and DINOv2 embedders.

        Args:
            cache_dir: Optional cache directory for model weights
        """
        self._cache_dir = cache_dir

        try:
            self._clip = CLIPEmbedder(cache_dir=cache_dir)
            self._dinov2 = DINOv2Embedder(cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize hybrid embedder: {e}") from e

    def encode_for_indexing(self, image: Image.Image) -> dict[str, list[float]]:
        """Encode image with both models for indexing.

        Args:
            image: PIL Image to encode

        Returns:
            Dictionary with 'clip' and 'dinov2' embeddings
        """
        return {
            "clip": self._clip.encode_image(image),
            "dinov2": self._dinov2.encode(image),
        }

    def batch_encode(self, images: list[Image.Image]) -> dict[str, list[list[float]]]:
        """Batch encode images with both models.

        Args:
            images: List of PIL Images to encode

        Returns:
            Dictionary with 'clip' and 'dinov2' batch embeddings
        """
        return {
            "clip": self._clip.batch_encode_images(images),
            "dinov2": self._dinov2.batch_encode(images),
        }

    def encode_query(self, query: str | Image.Image) -> tuple[str, list[float]]:
        """Encode query (text or image) and return appropriate vector.

        Args:
            query: Text query or PIL Image

        Returns:
            Tuple of (vector_name, embedding) where vector_name is 'clip' or 'dinov2'
        """
        if isinstance(query, str):
            # Text queries use CLIP's text encoder
            return ("clip", self._clip.encode_text(query))
        elif isinstance(query, Image.Image):
            # Image queries use DINOv2 for visual similarity
            return ("dinov2", self._dinov2.encode(query))
        else:
            raise TypeError(f"Query must be str or PIL Image, got {type(query)}")

    def get_model_versions(self) -> dict[str, str]:
        """Return version information for both models.

        Returns:
            Dictionary with 'clip' and 'dinov2' model versions
        """
        return {
            "clip": self._clip.model_version,
            "dinov2": self._dinov2.model_version,
        }
