"""CLIP-based description generator."""

from typing import Any, cast

import numpy as np

try:
    import PIL.Image
    import torch
    from transformers import CLIPModel, CLIPProcessor

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

from .base import DescriptionGenerator


class CLIPDescriptionGenerator(DescriptionGenerator):
    """Generate descriptions using CLIP vision-language model.

    CLIP can match images with text descriptions, enabling zero-shot
    classification and description generation.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        """Initialize CLIP generator.

        Args:
            model_name: HuggingFace model identifier for CLIP
        """
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self.candidate_descriptions: list[str] | None = None

        if HAS_CLIP:
            self._load_model()
            self._setup_default_descriptions()

    def _load_model(self):
        """Load CLIP model and processor."""
        if not HAS_CLIP:
            return

        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)

            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.model is not None:
                self.model = self.model.to(self.device)
                self.model.eval()

        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            self.model = None
            self.processor = None

    def _setup_default_descriptions(self):
        """Setup default candidate descriptions for zero-shot classification."""
        self.candidate_descriptions = [
            # UI Elements
            "a button",
            "a text input field",
            "a checkbox",
            "a radio button",
            "a dropdown menu",
            "a slider",
            "a toggle switch",
            "a progress bar",
            "a navigation menu",
            "a toolbar",
            "a sidebar",
            "a dialog box",
            "a modal window",
            "a tooltip",
            "a tab",
            "a scrollbar",
            # Content
            "an icon",
            "a logo",
            "an image",
            "a graph or chart",
            "a table",
            "a list",
            "a heading",
            "a paragraph of text",
            "a link",
            "a video player",
            # Containers
            "a panel",
            "a card",
            "a section",
            "a header",
            "a footer",
            # States/Actions
            "a close button",
            "a minimize button",
            "a maximize button",
            "a submit button",
            "a cancel button",
            "a save button",
            "a delete button",
            "a search box",
            "an error message",
            "a warning message",
            "a success message",
            "a loading indicator",
        ]

    def set_candidate_descriptions(self, descriptions: list[str]):
        """Set custom candidate descriptions for zero-shot classification.

        Args:
            descriptions: List of candidate descriptions
        """
        self.candidate_descriptions = descriptions

    def generate(
        self,
        image: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any] | None = None,
        bbox: tuple[Any, ...] | None = None,
    ) -> str:
        """Generate description using CLIP.

        Args:
            image: Input image (BGR format)
            mask: Optional mask
            bbox: Optional bounding box

        Returns:
            Best matching description
        """
        if not self.is_available() or not self.candidate_descriptions:
            return "unknown object"

        # Preprocess image
        region = self.preprocess_image(image, mask, bbox)

        # Convert BGR to RGB
        if len(region.shape) == 3 and region.shape[2] == 3:
            region = region[:, :, ::-1]

        # Convert to PIL Image
        pil_image = PIL.Image.fromarray(region.astype(np.uint8))

        try:
            # Prepare inputs
            if self.processor is None or self.device is None:
                return "unknown object"

            inputs = self.processor(
                text=self.candidate_descriptions,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get best match
            best_idx = probs.argmax().item()
            confidence = probs[0, best_idx].item()

            # Add confidence if high
            description = self.candidate_descriptions[best_idx]
            if confidence > 0.7:
                description = f"{description} (high confidence)"
            elif confidence < 0.3:
                description = f"{description} (uncertain)"

            return cast(str, description)

        except Exception as e:
            print(f"CLIP generation failed: {e}")
            return "unknown object"

    def batch_generate(
        self, image: np.ndarray[Any, Any], regions: list[Any]
    ) -> list[str]:
        """Generate descriptions for multiple regions.

        Args:
            image: Full image
            regions: List of regions with mask/bbox

        Returns:
            List of descriptions
        """
        if not self.is_available():
            return ["unknown object"] * len(regions)

        descriptions = []

        # Process in batches for efficiency
        batch_size = 8
        for i in range(0, len(regions), batch_size):
            batch = regions[i : i + batch_size]

            # Prepare batch of images
            pil_images = []
            for region_data in batch:
                mask = region_data.get("mask")
                bbox = region_data.get("bbox")
                region = self.preprocess_image(image, mask, bbox)

                # Convert BGR to RGB
                if len(region.shape) == 3 and region.shape[2] == 3:
                    region = region[:, :, ::-1]

                pil_images.append(PIL.Image.fromarray(region.astype(np.uint8)))

            try:
                if self.processor is None or self.model is None or self.device is None:
                    raise RuntimeError("Model not initialized")

                # Process batch
                inputs = self.processor(
                    text=self.candidate_descriptions,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Get best matches for each image
                for j in range(len(pil_images)):
                    best_idx = probs[j].argmax().item()
                    confidence = probs[j, best_idx].item()

                    description = self.candidate_descriptions[best_idx]
                    if confidence > 0.7:
                        description = f"{description} (high confidence)"
                    elif confidence < 0.3:
                        description = f"{description} (uncertain)"

                    descriptions.append(description)

            except Exception as e:
                print(f"Batch CLIP generation failed: {e}")
                descriptions.extend(["unknown object"] * len(batch))

        return descriptions

    def generate_free_form(
        self,
        image: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any] | None = None,
        bbox: tuple[Any, ...] | None = None,
    ) -> str:
        """Generate free-form description using image captioning.

        Note: This would require a different model like BLIP or GPT-4V.
        CLIP alone doesn't generate free-form captions.

        Args:
            image: Input image
            mask: Optional mask
            bbox: Optional bounding box

        Returns:
            Free-form description
        """
        # For now, fall back to classification
        return self.generate(image, mask, bbox)

    def is_available(self) -> bool:
        """Check if CLIP is available.

        Returns:
            True if CLIP is loaded and ready
        """
        return HAS_CLIP and self.model is not None and self.processor is not None

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Model identifier
        """
        return self.model_name
