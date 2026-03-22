"""QATM (Quality-Aware Template Matching) core algorithm.

Self-contained port of the QATM algorithm (kamata1729/QATM_pytorch).
Uses VGG-19 deep features with quality-aware scoring to evaluate how
unambiguous a template match is — rejecting false positives that
OpenCV matchTemplate would accept.

Reference: "QATM: Quality-Aware Template Matching For Deep Learning"
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .qatm_config import QATMSettings

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class QATMMatch:
    """A single QATM match result."""

    x: int
    y: int
    width: int
    height: int
    confidence: float


class QATMMatcher:
    """Quality-Aware Template Matching using VGG-19 deep features.

    Extracts feature maps from both template and screenshot using a
    pre-trained VGG-19, then computes a quality-aware similarity score
    that measures both match strength AND match uniqueness.

    Args:
        settings: QATM configuration. If None, uses defaults.
    """

    # VGG-19 layer name -> index in features Sequential
    _LAYER_MAP = {
        "relu1_1": 1,
        "relu1_2": 3,
        "relu2_1": 6,
        "relu2_2": 8,
        "relu3_1": 11,
        "relu3_2": 13,
        "relu3_3": 15,
        "relu3_4": 17,
        "relu4_1": 20,
        "relu4_2": 22,
        "relu4_3": 24,
        "relu4_4": 26,
        "relu5_1": 29,
        "relu5_2": 31,
        "relu5_3": 33,
        "relu5_4": 35,
    }

    def __init__(self, settings: QATMSettings | None = None) -> None:
        self._settings = settings or QATMSettings()
        self._model: Any = None
        self._device: Any = None
        self._transform: Any = None
        self._lock = threading.Lock()
        self._last_used: float = 0.0

    def _ensure_model(self) -> None:
        """Lazy-load VGG-19 feature extractor."""
        if self._model is not None:
            return

        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        device_str = self._settings.resolve_device()
        self._device = torch.device(device_str)

        layer_name = self._settings.feature_layer
        if layer_name not in self._LAYER_MAP:
            raise ValueError(
                f"Unknown VGG-19 layer '{layer_name}'. " f"Valid: {sorted(self._LAYER_MAP.keys())}"
            )
        layer_idx = self._LAYER_MAP[layer_name]

        # Load VGG-19 and truncate to the target layer
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self._model = vgg.features[: layer_idx + 1]
        self._model.eval()
        self._model.to(self._device)

        # Freeze parameters — inference only
        for param in self._model.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info("QATM: loaded VGG-19 (up to %s) on %s", layer_name, device_str)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy image to normalized tensor for VGG-19."""

        # BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb)  # type: ignore[misc]
        result: torch.Tensor = tensor.unsqueeze(0).to(self._device)  # (1, 3, H, W)
        return result

    def _extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract feature map from image using truncated VGG-19."""
        import torch

        tensor = self._preprocess(image)
        with torch.no_grad():
            features = self._model(tensor)  # type: ignore[misc]
        result: torch.Tensor = features
        return result

    def _compute_qatm_score(
        self,
        template_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> np.ndarray:
        """Compute quality-aware template matching score map.

        The QATM score at each location measures both:
        1. How well the template matches at that location (strength)
        2. How much the match stands out from other locations (quality)

        Returns:
            2D score map (H, W) where H, W are the valid convolution output
            dimensions. Values in [0, 1].
        """
        import torch
        import torch.nn.functional as F

        alpha = self._settings.alpha

        # template_features: (1, C, Ht, Wt)
        # image_features:    (1, C, Hi, Wi)
        t_feat = template_features.squeeze(0)  # (C, Ht, Wt)
        i_feat = image_features.squeeze(0)  # (C, Hi, Wi)

        c, ht, wt = t_feat.shape
        _, hi, wi = i_feat.shape

        # Flatten spatial dims: template -> (C, Ht*Wt), image -> (C, Hi*Wi)
        t_flat = t_feat.reshape(c, -1)  # (C, Nt)
        i_flat = i_feat.reshape(c, -1)  # (C, Ni)

        # Normalize features along channel dimension
        t_norm = F.normalize(t_flat, dim=0)  # (C, Nt)
        i_norm = F.normalize(i_flat, dim=0)  # (C, Ni)

        # Cosine similarity matrix: (Nt, Ni)
        cos_sim = torch.mm(t_norm.t(), i_norm)  # (Nt, Ni)

        # QATM quality-aware scoring:
        # For each template location, softmax across image locations (how
        # unique is the best match for this template patch?)
        # For each image location, softmax across template locations (how
        # well does this image patch match the full template?)
        qatm_t = F.softmax(alpha * cos_sim, dim=1)  # (Nt, Ni)
        qatm_i = F.softmax(alpha * cos_sim, dim=0)  # (Nt, Ni)

        # Combined quality score: element-wise product, sum over template dims
        # This produces a score for each image spatial location
        qatm_score = (qatm_t * qatm_i).sum(dim=0)  # (Ni,)

        # Reshape back to spatial dimensions
        score_map = qatm_score.reshape(hi, wi).cpu().numpy()

        # Resize score map to match the valid correlation output size
        out_h = hi
        out_w = wi
        if out_h <= 0 or out_w <= 0:
            return np.zeros((1, 1), dtype=np.float32)

        # Normalize to [0, 1]
        score_min = score_map.min()
        score_max = score_map.max()
        if score_max - score_min > 1e-8:
            score_map = (score_map - score_min) / (score_max - score_min)
        else:
            score_map = np.zeros_like(score_map)

        return score_map.astype(np.float32)

    def find(
        self,
        template: np.ndarray,
        screenshot: np.ndarray,
        min_confidence: float = 0.7,
        find_all: bool = False,
    ) -> list[QATMMatch]:
        """Find template in screenshot using quality-aware matching.

        Args:
            template: Template image as BGR numpy array.
            screenshot: Screenshot image as BGR numpy array.
            min_confidence: Minimum quality-aware confidence (0.0-1.0).
            find_all: If True, return all matches above threshold.
                      If False, return only the best match.

        Returns:
            List of QATMMatch sorted by confidence (highest first).
        """
        with self._lock:
            self._ensure_model()
            self._last_used = time.monotonic()

        t_h, t_w = template.shape[:2]
        s_h, s_w = screenshot.shape[:2]

        if t_h > s_h or t_w > s_w:
            logger.debug("QATM: template larger than screenshot, skipping")
            return []

        # Extract deep features
        with self._lock:
            t_features = self._extract_features(template)
            s_features = self._extract_features(screenshot)

        # Compute quality-aware score map
        score_map = self._compute_qatm_score(t_features, s_features)

        # Scale factor: feature map is smaller than input image due to
        # VGG pooling layers. We need to map feature-space coordinates
        # back to pixel-space.
        scale_y = s_h / score_map.shape[0]
        scale_x = s_w / score_map.shape[1]

        results: list[QATMMatch] = []

        if find_all:
            # Find all peaks above threshold
            locations = np.where(score_map >= min_confidence)
            for fy, fx in zip(locations[0], locations[1], strict=True):
                confidence = float(score_map[fy, fx])
                px = int(fx * scale_x)
                py = int(fy * scale_y)
                results.append(
                    QATMMatch(
                        x=max(0, px - t_w // 2),
                        y=max(0, py - t_h // 2),
                        width=t_w,
                        height=t_h,
                        confidence=confidence,
                    )
                )
            # Apply NMS to remove overlapping detections
            results = self._nms(results, iou_threshold=0.5)
        else:
            # Return only the best match
            max_val = float(score_map.max())
            if max_val >= min_confidence:
                max_loc = np.unravel_index(score_map.argmax(), score_map.shape)
                fy, fx = max_loc
                px = int(fx * scale_x)
                py = int(fy * scale_y)
                results.append(
                    QATMMatch(
                        x=max(0, px - t_w // 2),
                        y=max(0, py - t_h // 2),
                        width=t_w,
                        height=t_h,
                        confidence=max_val,
                    )
                )

        results.sort(key=lambda m: m.confidence, reverse=True)
        return results

    @staticmethod
    def _nms(matches: list[QATMMatch], iou_threshold: float = 0.5) -> list[QATMMatch]:
        """Non-Maximum Suppression to remove overlapping detections."""
        if not matches:
            return []

        # Sort by confidence descending
        matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        kept: list[QATMMatch] = []

        for candidate in matches:
            overlaps = False
            for existing in kept:
                iou = _compute_iou(candidate, existing)
                if iou > iou_threshold:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(candidate)

        return kept

    def unload(self) -> None:
        """Release the VGG-19 model from memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._device = None

                # Try to free GPU memory
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                logger.info("QATM: model unloaded")

    def should_unload(self) -> bool:
        """Check if the model should be unloaded due to inactivity."""
        timeout = self._settings.unload_after_seconds
        if timeout <= 0 or self._model is None:
            return False
        return (time.monotonic() - self._last_used) > timeout


def _compute_iou(a: QATMMatch, b: QATMMatch) -> float:
    """Compute Intersection over Union between two matches."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
