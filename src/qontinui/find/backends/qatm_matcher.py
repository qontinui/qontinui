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
    ) -> tuple[np.ndarray, int, int]:
        """Compute quality-aware template matching score map.

        Uses sliding-window correlation (F.conv2d) to produce a proper
        valid-convolution output, then applies QATM quality-aware scoring
        to measure both match strength AND uniqueness.

        Returns:
            Tuple of (score_map, feat_ht, feat_wt) where score_map has
            shape (Hi-Ht+1, Wi-Wt+1) with values in [0, 1], and feat_ht/wt
            are the template feature dimensions (needed for coordinate mapping).
        """
        import torch.nn.functional as F

        alpha = self._settings.alpha

        # template_features: (1, C, Ht, Wt)
        # image_features:    (1, C, Hi, Wi)
        c = template_features.shape[1]
        ht, wt = template_features.shape[2], template_features.shape[3]

        # Normalize features along channel dimension
        t_norm = F.normalize(template_features, dim=1)  # (1, C, Ht, Wt)
        i_norm = F.normalize(image_features, dim=1)  # (1, C, Hi, Wi)

        # Sliding-window cosine similarity via conv2d.
        # Reshape template as C conv filters of size (Ht, Wt), each with 1 channel.
        # Use groups=C to compute per-channel correlation, then sum.
        # This produces an output of valid size: (Hi-Ht+1, Wi-Wt+1).
        t_kernel = t_norm.squeeze(0).unsqueeze(1)  # (C, 1, Ht, Wt)
        # Per-channel sliding dot product, then sum across channels
        per_channel = F.conv2d(i_norm, t_kernel, groups=c)  # (1, C, Ho, Wo)
        # Sum across channels and average by template spatial size
        # to get mean cosine similarity per sliding position
        cos_sim_map = per_channel.sum(dim=1, keepdim=True) / (ht * wt)  # (1, 1, Ho, Wo)

        # Flatten to (N_positions,) for QATM quality scoring
        ho, wo = cos_sim_map.shape[2], cos_sim_map.shape[3]
        cos_sim_flat = cos_sim_map.reshape(-1)  # (Ho*Wo,)

        # QATM quality-aware scoring:
        # Apply softmax across all positions — measures how much the best
        # match stands out from alternatives. A unique match gets high
        # quality; ambiguous matches (many similar buttons) get low quality.
        qatm_score = F.softmax(alpha * cos_sim_flat, dim=0)  # (Ho*Wo,)

        # Reshape back to spatial dimensions
        score_map = qatm_score.reshape(ho, wo).cpu().numpy()

        if score_map.size == 0:
            return np.zeros((1, 1), dtype=np.float32), ht, wt

        # Normalize to [0, 1]
        score_min = score_map.min()
        score_max = score_map.max()
        if score_max - score_min > 1e-8:
            score_map = (score_map - score_min) / (score_max - score_min)
        else:
            score_map = np.zeros_like(score_map)

        return score_map.astype(np.float32), ht, wt

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
        t_h, t_w = template.shape[:2]
        s_h, s_w = screenshot.shape[:2]

        if t_h > s_h or t_w > s_w:
            logger.debug("QATM: template larger than screenshot, skipping")
            return []

        # Hold lock for entire GPU computation to prevent race conditions
        # with concurrent unload() calls or parallel find() on same device.
        with self._lock:
            self._ensure_model()
            self._last_used = time.monotonic()

            t_features = self._extract_features(template)
            s_features = self._extract_features(screenshot)
            score_map, feat_ht, feat_wt = self._compute_qatm_score(t_features, s_features)

        # Score map has shape (Ho, Wo) = valid convolution output.
        # Each position (fy, fx) corresponds to where the template feature
        # block starts. Map feature-space back to pixel-space for the
        # top-left corner of the matched region.
        #
        # VGG pooling reduces spatial dims. The scale factor maps feature
        # positions to pixel positions of the receptive field top-left.
        # The score map is already "valid" (no border artifacts), so
        # position (fy, fx) maps to pixel top-left at (fy * scale, fx * scale).
        ho, wo = score_map.shape
        if ho == 0 or wo == 0:
            return []

        # Compute scale: image pixels per feature-map cell
        # Use the full feature map size (ho + feat_ht - 1) to recover
        # the image-to-feature ratio, since ho = Hi - Ht + 1.
        feat_hi = ho + feat_ht - 1
        feat_wi = wo + feat_wt - 1
        scale_y = s_h / feat_hi
        scale_x = s_w / feat_wi

        results: list[QATMMatch] = []

        if find_all:
            locations = np.where(score_map >= min_confidence)
            for fy, fx in zip(locations[0], locations[1], strict=True):
                confidence = float(score_map[fy, fx])
                px = int(fx * scale_x)
                py = int(fy * scale_y)
                results.append(
                    QATMMatch(
                        x=max(0, px),
                        y=max(0, py),
                        width=t_w,
                        height=t_h,
                        confidence=confidence,
                    )
                )
            results = self._nms(results, iou_threshold=0.5)
        else:
            max_val = float(score_map.max())
            if max_val >= min_confidence:
                max_loc = np.unravel_index(score_map.argmax(), score_map.shape)
                fy, fx = max_loc
                px = int(fx * scale_x)
                py = int(fy * scale_y)
                results.append(
                    QATMMatch(
                        x=max(0, px),
                        y=max(0, py),
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
