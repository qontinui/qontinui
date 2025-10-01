"""Element vectorization using vision models."""

from typing import Any, cast

import cv2
import numpy as np
from PIL import Image


class ObjectVectorizer:
    """Vectorize UI elements using vision models for semantic matching."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize ObjectVectorizer.

        Args:
            model_name: Name of the vision model to use
        """
        self.model_name = model_name
        self.clip_model = None
        self.clip_processor = None
        self.device = "cpu"

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the vision model."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self.clip_model = CLIPModel.from_pretrained(self.model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)

            # Use GPU if available
            if torch.cuda.is_available():
                self.device = "cuda"
                self.clip_model = self.clip_model.to(self.device)

            print(f"CLIP model initialized: {self.model_name} on {self.device}")
        except ImportError:
            print("Transformers library not available, using fallback vectorization")
        except Exception as e:
            print(f"Failed to initialize CLIP model: {e}")

    def vectorize_element(self, element_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate embedding vector for a UI element.

        Args:
            element_image: Element image as numpy array (BGR format)

        Returns:
            Embedding vector as numpy array
        """
        if self.clip_model is not None:
            return self._vectorize_with_clip(element_image)
        else:
            return self._vectorize_fallback(element_image)

    def _vectorize_with_clip(self, element_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Vectorize using CLIP model.

        Args:
            element_image: Element image as numpy array

        Returns:
            CLIP embedding vector
        """
        import torch

        # Convert BGR to RGB
        if len(element_image.shape) == 3 and element_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(element_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = element_image

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Process image
        if self.clip_processor is None:
            raise RuntimeError("CLIP processor not initialized")
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")

        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get image features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        # Normalize and convert to numpy
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().flatten()

        return embedding

    def _vectorize_fallback(self, element_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Fallback vectorization using traditional computer vision features.

        Args:
            element_image: Element image as numpy array

        Returns:
            Feature vector
        """
        features = []

        # Color histogram features
        hist_features = self._extract_color_histogram(element_image)
        features.extend(hist_features)

        # Edge features
        edge_features = self._extract_edge_features(element_image)
        features.extend(edge_features)

        # Texture features
        texture_features = self._extract_texture_features(element_image)
        features.extend(texture_features)

        # Shape features
        shape_features = self._extract_shape_features(element_image)
        features.extend(shape_features)

        return np.array(features, dtype=np.float32)

    def _extract_color_histogram(self, image: np.ndarray[Any, Any], bins: int = 32) -> list[float]:
        """Extract color histogram features.

        Args:
            image: Input image
            bins: Number of histogram bins

        Returns:
            Flattened histogram features
        """
        features = []

        # Calculate histograms for each channel
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist.tolist())

        return features

    def _extract_edge_features(self, image: np.ndarray[Any, Any]) -> list[float]:
        """Extract edge-based features.

        Args:
            image: Input image

        Returns:
            Edge features
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size

        # Horizontal and vertical edge projections
        h_projection = np.mean(edges, axis=1)
        v_projection = np.mean(edges, axis=0)

        features = [
            edge_density,
            np.mean(h_projection),
            np.std(h_projection),
            np.mean(v_projection),
            np.std(v_projection),
        ]

        return features

    def _extract_texture_features(self, image: np.ndarray[Any, Any]) -> list[float]:
        """Extract texture features using Gabor filters.

        Args:
            image: Input image

        Returns:
            Texture features
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []

        # Apply Gabor filters at different orientations
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)  # type: ignore[attr-defined]
            features.extend(
                [
                    float(np.mean(filtered)),
                    float(np.std(filtered)),
                ]
            )

        return features

    def _extract_shape_features(self, image: np.ndarray[Any, Any]) -> list[float]:
        """Extract shape-based features.

        Args:
            image: Input image

        Returns:
            Shape features
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h if h > 0 else 0

        # Calculate image moments
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)

        # Hu moments (shape descriptors)
        hu_moments = cv2.HuMoments(moments).flatten()

        features = [aspect_ratio] + hu_moments.tolist()[:7]

        return cast(list[float], features)

    def vectorize_batch(self, images: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Vectorize a batch of images.

        Args:
            images: List of images as numpy arrays

        Returns:
            Array of embedding vectors
        """
        embeddings = []
        for image in images:
            embedding = self.vectorize_element(image)
            embeddings.append(embedding)

        return np.array(embeddings)

    def compute_similarity(
        self, embedding1: np.ndarray[Any, Any], embedding2: np.ndarray[Any, Any]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2

        # Compute cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)

        # Ensure in [0, 1] range
        return float(np.clip(similarity, 0, 1))

    def encode_text(self, text: str | list[str]) -> np.ndarray[Any, Any]:
        """Encode text descriptions for cross-modal matching.

        Args:
            text: Text description(s) to encode

        Returns:
            Text embedding(s)
        """
        if self.clip_model is None:
            # Return random embedding as fallback
            return np.random.randn(512)

        import torch

        # Ensure text is a list
        if isinstance(text, str):
            text = [text]

        # Process text
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)

        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get text features
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)

        # Normalize and convert to numpy
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embeddings = text_features.cpu().numpy()

        if len(text) == 1:
            return embeddings[0]
        return embeddings
