"""
EXPERIMENTAL: SAM3 (Segment Anything Model 3) based detection.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

SAM3 is a cutting-edge image segmentation model that can be used for
automatic GUI element detection. This implementation is experimental and
requires additional dependencies.

Dependencies:
    - sam3: pip install git+https://github.com/facebookresearch/sam3.git
    - PyTorch 2.7+: pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    - Python 3.12+
    - CUDA 12.6+

Note: SAM3 requires access to model checkpoints which may require registration.
"""

import os
from typing import Any

import cv2
import numpy as np

from .base_detector import BaseDetector
from .types import BBox


class SAM3Detector(BaseDetector):
    """Detect GUI elements using SAM3 (Segment Anything Model 3).

    EXPERIMENTAL: This detector uses the experimental SAM3 model for
    automatic GUI element segmentation. The model and API are subject to change.

    This detector can perform:
    - Automatic segmentation using grid-based point prompts
    - Text-guided concept-based segmentation
    - Interactive segmentation from user prompts

    Note: Requires SAM3 checkpoint files which may require registration
    on Hugging Face or the SAM3 repository.
    """

    def __init__(self):
        super().__init__("SAM3 Detector")
        self.sam_available = False
        self.processor = None
        self._try_load_sam3()

    def _try_load_sam3(self):
        """Try to load SAM3 model"""
        try:
            import torch
            from sam3.build_sam import build_sam3_image_model

            self.sam_available = True

            # Check for model checkpoint
            checkpoint_path = None

            # Common checkpoint paths for SAM3
            possible_paths = [
                "checkpoints/sam3_hiera_large.pt",
                "sam3_hiera_large.pt",
                os.path.expanduser("~/.cache/sam3/sam3_hiera_large.pt"),
                os.path.expanduser(
                    "~/.cache/huggingface/hub/models--facebook--sam3-hiera-large/pytorch_model.bin"
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break

            if checkpoint_path is None:
                print("SAM3 checkpoint not found. Will skip SAM3 detection.")
                print("Download from: https://github.com/facebookresearch/sam3")
                print("Note: You need to request access to SAM3 checkpoints on Hugging Face")
                self.sam_available = False
                return

            # Build SAM3 model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sam3_model = build_sam3_image_model(checkpoint_path, device=device)

            # Import Sam3Processor for handling images
            from sam3 import Sam3Processor

            self.processor = Sam3Processor(self.sam3_model)

            print(f"SAM3 loaded successfully on {device}")

        except ImportError as e:
            print(f"SAM3 not available: {e}")
            print("Install with: pip install git+https://github.com/facebookresearch/sam3.git")
            self.sam_available = False
        except Exception as e:
            print(f"Error loading SAM3: {e}")
            self.sam_available = False

    def detect(self, image_path: str, **params) -> list[BBox]:
        """
        Detect elements using SAM3

        Parameters:
            text_prompt: Optional text description for concept-based segmentation
            min_area: Minimum mask area (default: 100)
            grid_points: Number of grid points for automatic segmentation (default: 32)
        """
        if not self.sam_available or self.processor is None:
            print("SAM3 not available, skipping...")
            return []

        img = cv2.imread(image_path)
        if img is None:
            return []

        # Convert BGR to RGB and to PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image

        pil_img = Image.fromarray(img_rgb)

        # Parameters
        text_prompt = params.get("text_prompt", None)
        min_area = params.get("min_area", 100)
        max_area = params.get("max_area", img.shape[0] * img.shape[1] * 0.9)
        grid_points = params.get("grid_points", 32)

        try:
            # Set image for processing
            self.processor.set_image(pil_img)

            boxes = []

            if text_prompt:
                # Use text-based concept segmentation (SAM3's new feature)
                self.processor.set_text_prompt(text_prompt)
                results = self.processor.segment()

                if results and "masks" in results:
                    for idx, _mask in enumerate(results["masks"]):
                        bbox = results.get("boxes", [])[idx] if "boxes" in results else None
                        confidence = results.get("scores", [])[idx] if "scores" in results else 1.0

                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            area = (x2 - x1) * (y2 - y1)
                            if area >= min_area and area <= max_area:
                                box = BBox(
                                    int(x1),
                                    int(y1),
                                    int(x2),
                                    int(y2),
                                    confidence=float(confidence),
                                )
                                boxes.append(box)
            else:
                # Automatic segmentation using grid of points (similar to SAM2's automatic mode)
                # Generate a grid of points for automatic segmentation
                h, w = img.shape[:2]
                step_x = w // grid_points
                step_y = h // grid_points

                for i in range(grid_points):
                    for j in range(grid_points):
                        x = step_x * i + step_x // 2
                        y = step_y * j + step_y // 2

                        # Use point prompt for segmentation
                        results = self.processor.segment_from_point(x, y)

                        if results and "masks" in results and len(results["masks"]) > 0:
                            mask = results["masks"][0]

                            # Get bounding box from mask
                            y_indices, x_indices = np.where(mask)
                            if len(x_indices) == 0 or len(y_indices) == 0:
                                continue

                            x1, x2 = int(x_indices.min()), int(x_indices.max())
                            y1, y2 = int(y_indices.min()), int(y_indices.max())

                            area = (x2 - x1) * (y2 - y1)
                            if area >= min_area and area <= max_area:
                                confidence = results.get("scores", [1.0])[0]
                                box = BBox(x1, y1, x2, y2, confidence=float(confidence))
                                boxes.append(box)

            # Post-process to remove duplicates
            boxes = self.remove_contained_boxes(boxes)
            boxes = self.merge_overlapping_boxes(boxes, iou_threshold=0.8)

            return boxes

        except Exception as e:
            print(f"Error during SAM3 detection: {e}")
            import traceback

            traceback.print_exc()
            return []

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Parameter grid for hyperparameter search"""
        if not self.sam_available:
            return []

        return [
            # Fine-grained detection
            {"grid_points": 32, "min_area": 100},
            {"grid_points": 48, "min_area": 100},
            # Coarse detection
            {"grid_points": 24, "min_area": 100},
            {"grid_points": 16, "min_area": 150},
            # Text-based concept segmentation examples
            {"text_prompt": "button", "min_area": 100},
            {"text_prompt": "text input field", "min_area": 100},
            {"text_prompt": "icon", "min_area": 50},
            {"text_prompt": "menu item", "min_area": 100},
        ]
