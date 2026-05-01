"""Scale-adaptive template matching detection backend.

Port of the robust template-matching technique introduced by
Kamata & Tsumura (APSIPA 2017), as implemented in
https://github.com/kamata1729/robustTemplateMatching.  The core idea:
extract a deep feature map (VGG-13 features up to ``relu4_1``) for both
template and screenshot, evaluate the template against the screenshot
at several template scales, and pick the best match across scales.

This tier sits in the cascade between Invariant Match (~120ms) and
QATM (~200ms) at ~140ms, targeting UI targets that survived pixel-level
template matching but are rendered at a different size than the
captured template — typical symptoms are DPI changes and responsive
layout reflow.

Rollout-safe: the backend is gated by ``QONTINUI_ENABLE_SCALE_ADAPTIVE_MATCH``.
When the flag is unset (the default), ``supports()`` returns False for
every needle type and the cascade skips the backend.  No model or
torchvision import happens in that path.

TODO(grounding-v2 Phase 1b): the upstream implementation ships a Cython
extension that fuses the feature-normalise / sliding-correlation loop
for a ~2x speedup on CPU.  This pass is pure Python + PyTorch for
portability; the Cython port is tracked as a follow-up optimisation
once the tier proves useful on the eval harness.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._feature_cache import get_feature_cache, hash_screenshot
from .base import DetectionBackend, DetectionResult

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

ENV_FLAG = "QONTINUI_ENABLE_SCALE_ADAPTIVE_MATCH"

# VGG-13 ``features`` Sequential layer indices for the post-ReLU
# activations we care about.  Indices are stable across torchvision
# versions — they come from the canonical Conv/ReLU/Conv/ReLU/Pool
# structure of VGG-13 (``models.vgg13``).
_VGG13_LAYER_MAP = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu4_1": 16,
    "relu4_2": 18,
    "relu5_1": 21,
    "relu5_2": 23,
}

_DEFAULT_LAYER = "relu4_1"
# Scale factors applied to the template before feature extraction.
# Chosen to cover the common DPI / layout-reflow range (~70% .. ~140%)
# without blowing up runtime; 5 scales fits the ~140ms budget on GPU.
_DEFAULT_SCALES: tuple[float, ...] = (0.70, 0.85, 1.00, 1.18, 1.40)
_UNLOAD_AFTER_SECONDS = 300.0
_BACKEND_NAME = "scale_adaptive"
_CACHE_KEY_PREFIX = "scale_adaptive_vgg13"


def is_enabled() -> bool:
    """Whether the env gate is set to '1'.

    Anything other than exactly ``"1"`` keeps the tier disabled — we
    intentionally do not accept "true"/"yes" here.  The spec calls the
    gate a "rollout safety valve", and a strict comparison makes it
    obvious in logs and CI that the tier is off.
    """
    return os.environ.get(ENV_FLAG, "0") == "1"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


def _ensure_bgr_array(arr: np.ndarray) -> np.ndarray:
    """Ensure a numpy array is 3-channel BGR (same semantics as qatm_backend)."""
    import cv2

    if len(arr.shape) == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if len(arr.shape) == 3:
        if arr.shape[2] == 1:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] >= 3:
            return arr[:, :, :3]
    return arr


@dataclass
class _ScaleMatch:
    """An internal match returned by the inner matching routine."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    scale: float


class _VGG13FeatureExtractor:
    """Lazy-loaded VGG-13 feature extractor with idle unload.

    Mirrors the QATM matcher's ``_ensure_model`` pattern: hold a lock
    over GPU work so a concurrent ``unload()`` can't yank tensors out
    from under a running ``forward``.
    """

    def __init__(self, layer: str = _DEFAULT_LAYER) -> None:
        if layer not in _VGG13_LAYER_MAP:
            raise ValueError(
                f"Unknown VGG-13 layer '{layer}'. Valid: {sorted(_VGG13_LAYER_MAP.keys())}"
            )
        self._layer = layer
        self._model: Any = None
        self._device: Any = None
        self._transform: Any = None
        self._lock = threading.Lock()
        self._last_used: float = 0.0

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)

        # models.vgg13 with pretrained weights — same API the upstream
        # reference uses (``models.vgg13(pretrained=True).features``).
        vgg = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        layer_idx = _VGG13_LAYER_MAP[self._layer]
        self._model = vgg.features[: layer_idx + 1]
        self._model.eval()
        self._model.to(self._device)

        for param in self._model.parameters():
            param.requires_grad = False

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info(
            "ScaleAdaptive: loaded VGG-13 (up to %s) on %s", self._layer, device_str
        )

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb)  # type: ignore[misc]
        result: torch.Tensor = tensor.unsqueeze(0).to(self._device)
        return result

    def extract(self, bgr: np.ndarray) -> torch.Tensor:
        """Run VGG forward pass on a BGR numpy image.

        Must be called while holding ``self._lock`` if the caller also
        orchestrates unload timing.
        """
        import torch

        self._ensure_model()
        self._last_used = time.monotonic()
        tensor = self._preprocess(bgr)
        with torch.no_grad():
            features: torch.Tensor = self._model(tensor)  # type: ignore[misc]
        return features

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def touch(self) -> None:
        self._last_used = time.monotonic()

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            del self._model
            self._model = None
            self._device = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("ScaleAdaptive: model unloaded")

    def should_unload(self) -> bool:
        if self._model is None:
            return False
        return (time.monotonic() - self._last_used) > _UNLOAD_AFTER_SECONDS


class ScaleAdaptiveBackend(DetectionBackend):
    """Scale-adaptive deep template matching (robust-TM, APSIPA 2017).

    Probes several template scales against a cached deep feature map
    of the screenshot.  Robust to the resolution/DPI/layout drift that
    defeats pixel-exact and invariant template matching.

    Enable via ``QONTINUI_ENABLE_SCALE_ADAPTIVE_MATCH=1``.  When unset,
    the backend is silently inert — ``supports()`` returns False for
    every needle type so the cascade skips it without loading any
    model.  This is the intended default while the tier is under
    rollout.
    """

    def __init__(
        self,
        *,
        layer: str = _DEFAULT_LAYER,
        scales: tuple[float, ...] = _DEFAULT_SCALES,
    ) -> None:
        self._layer = layer
        self._scales = tuple(scales)
        self._extractor: _VGG13FeatureExtractor | None = None

    # ------------------------------------------------------------------
    # DetectionBackend ABC
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return _BACKEND_NAME

    def estimated_cost_ms(self) -> float:
        return 140.0

    def supports(self, needle_type: str) -> bool:
        if not is_enabled():
            return False
        return needle_type == "template"

    def is_available(self) -> bool:
        return is_enabled() and _torch_available()

    def find(
        self, needle: Any, haystack: Any, config: dict[str, Any]
    ) -> list[DetectionResult]:
        if not is_enabled():
            return []

        template = self._to_bgr(needle)
        screenshot = self._to_bgr(haystack)
        if template is None or screenshot is None:
            logger.debug(
                "ScaleAdaptiveBackend: could not convert needle/haystack to BGR"
            )
            return []

        # Apply search-region cropping if requested.
        search_region = config.get("search_region")
        region_offset_x, region_offset_y = 0, 0
        if search_region is not None:
            rx, ry, rw, rh = search_region
            sh, sw = screenshot.shape[:2]
            rx = max(0, min(rx, sw))
            ry = max(0, min(ry, sh))
            rw = min(rw, sw - rx)
            rh = min(rh, sh - ry)
            if rw > 0 and rh > 0:
                screenshot = screenshot[ry : ry + rh, rx : rx + rw]
                region_offset_x, region_offset_y = rx, ry

        min_confidence = config.get("min_confidence", 0.7)
        find_all = bool(config.get("find_all", False))
        # Optional per-call override — useful for the eval harness.
        scales = tuple(config.get("scale_adaptive_scales", self._scales))

        extractor = self._get_extractor()

        try:
            matches = self._run_matching(
                extractor=extractor,
                template=template,
                screenshot=screenshot,
                scales=scales,
                min_confidence=min_confidence,
                find_all=find_all,
            )
        except Exception:
            logger.exception("ScaleAdaptiveBackend: matching failed")
            return []
        finally:
            if extractor.should_unload():
                extractor.unload()

        results: list[DetectionResult] = []
        for m in matches:
            results.append(
                DetectionResult(
                    x=m.x + region_offset_x,
                    y=m.y + region_offset_y,
                    width=m.width,
                    height=m.height,
                    confidence=m.confidence,
                    backend_name=self.name,
                    metadata={"scale": m.scale, "layer": self._layer},
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_extractor(self) -> _VGG13FeatureExtractor:
        if self._extractor is None:
            self._extractor = _VGG13FeatureExtractor(layer=self._layer)
        return self._extractor

    def _run_matching(
        self,
        *,
        extractor: _VGG13FeatureExtractor,
        template: np.ndarray,
        screenshot: np.ndarray,
        scales: tuple[float, ...],
        min_confidence: float,
        find_all: bool,
    ) -> list[_ScaleMatch]:
        """Multi-scale VGG correlation matching.

        Algorithm (matches the upstream reference's control flow):
          1. Extract the screenshot feature map once.  Cache it keyed
             by ``blake2b(screenshot.tobytes())`` so later cascade
             calls against the same screenshot skip the forward pass.
          2. For each scale factor s, rescale the template, extract
             its feature map, normalise both feature maps along the
             channel dimension, and run a valid-mode grouped conv2d
             to get the sliding cosine-similarity score map.
          3. Track the best score across scales.  In ``find_all`` mode
             gather every local max above threshold; otherwise just
             the single best match.
        """
        import cv2
        import torch
        import torch.nn.functional as F  # noqa: N812

        s_h, s_w = screenshot.shape[:2]
        t_h, t_w = template.shape[:2]

        if t_h == 0 or t_w == 0 or s_h == 0 or s_w == 0:
            return []

        feature_cache = get_feature_cache()
        screenshot_hash = hash_screenshot(screenshot)
        cache_key = f"{_CACHE_KEY_PREFIX}_{self._layer}"

        # Hold the extractor lock over every GPU op to interlock with
        # unload().  The lock is re-entrant-free, but we never call
        # back into ``find`` from within this block.
        with extractor.lock:
            cached = feature_cache.get(screenshot_hash, cache_key)
            if cached is not None:
                s_features = cached
                extractor.touch()
            else:
                s_features = extractor.extract(screenshot)
                feature_cache.put(screenshot_hash, cache_key, s_features)

            best: _ScaleMatch | None = None
            all_matches: list[_ScaleMatch] = []

            for scale in scales:
                # Skip scales that would produce a template larger than
                # the screenshot — the valid-mode conv would have no
                # output positions.
                scaled_w = max(1, int(round(t_w * scale)))
                scaled_h = max(1, int(round(t_h * scale)))
                if scaled_h > s_h or scaled_w > s_w:
                    continue

                if scale == 1.0:
                    scaled_template = template
                else:
                    scaled_template = cv2.resize(
                        template,
                        (scaled_w, scaled_h),
                        interpolation=(
                            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                        ),
                    )

                t_features = extractor.extract(scaled_template)

                # Feature-map shape guard: if the chosen layer pooled
                # the scaled template down to <=0 cells in either
                # dimension, nothing to correlate against.
                ht, wt = t_features.shape[2], t_features.shape[3]
                hi, wi = s_features.shape[2], s_features.shape[3]
                if ht > hi or wt > wi or ht == 0 or wt == 0:
                    continue

                t_norm = F.normalize(t_features, dim=1)
                i_norm = F.normalize(s_features, dim=1)

                c = t_norm.shape[1]
                t_kernel = t_norm.squeeze(0).unsqueeze(1)  # (C, 1, ht, wt)
                per_channel = F.conv2d(i_norm, t_kernel, groups=c)
                # Mean cosine similarity per sliding position, range [-1, 1].
                cos_sim_map = per_channel.sum(dim=1, keepdim=True) / (ht * wt)
                # Clamp in case of tiny numerical drift outside [-1, 1].
                cos_sim_map = torch.clamp(cos_sim_map, -1.0, 1.0)
                # Remap to [0, 1] — keeps the "confidence" convention
                # that the rest of the cascade expects.
                conf_map = ((cos_sim_map + 1.0) * 0.5).squeeze(0).squeeze(0)

                ho, wo = conf_map.shape
                if ho == 0 or wo == 0:
                    continue

                feat_hi = ho + ht - 1
                feat_wi = wo + wt - 1
                # s_features extracted from an (s_h, s_w) BGR image.
                # Map feature coords back to pixel coords using the
                # actual ratio (the pooling factor depends on the
                # chosen layer; computing it from shapes is robust).
                scale_y = s_h / feat_hi
                scale_x = s_w / feat_wi

                conf_np = conf_map.detach().cpu().numpy()

                if find_all:
                    hits = np.argwhere(conf_np >= min_confidence)
                    for fy, fx in hits:
                        confidence = float(conf_np[fy, fx])
                        px = int(fx * scale_x)
                        py = int(fy * scale_y)
                        all_matches.append(
                            _ScaleMatch(
                                x=max(0, px),
                                y=max(0, py),
                                width=scaled_w,
                                height=scaled_h,
                                confidence=confidence,
                                scale=scale,
                            )
                        )
                else:
                    max_val = float(conf_np.max())
                    if best is None or max_val > best.confidence:
                        fy, fx = np.unravel_index(conf_np.argmax(), conf_np.shape)
                        best = _ScaleMatch(
                            x=max(0, int(fx * scale_x)),
                            y=max(0, int(fy * scale_y)),
                            width=scaled_w,
                            height=scaled_h,
                            confidence=max_val,
                            scale=scale,
                        )

        if find_all:
            all_matches = _nms(all_matches, iou_threshold=0.5)
            all_matches.sort(key=lambda m: m.confidence, reverse=True)
            return all_matches

        if best is None or best.confidence < min_confidence:
            return []
        return [best]

    @staticmethod
    def _to_bgr(image: Any) -> np.ndarray | None:
        """Convert image to BGR numpy array.  Mirrors QATMBackend._to_bgr."""
        if isinstance(image, np.ndarray):
            return _ensure_bgr_array(image)

        if hasattr(image, "pixel_data") and image.pixel_data is not None:
            data = image.pixel_data
            if isinstance(data, np.ndarray):
                return _ensure_bgr_array(data)
            return ScaleAdaptiveBackend._to_bgr(data)

        try:
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                import cv2

                arr = np.array(image.convert("RGB"))
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        return None


def _nms(matches: list[_ScaleMatch], iou_threshold: float = 0.5) -> list[_ScaleMatch]:
    """Non-Maximum Suppression keyed on confidence."""
    if not matches:
        return []
    ordered = sorted(matches, key=lambda m: m.confidence, reverse=True)
    kept: list[_ScaleMatch] = []
    for cand in ordered:
        overlaps = False
        for existing in kept:
            if _iou(cand, existing) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(cand)
    return kept


def _iou(a: _ScaleMatch, b: _ScaleMatch) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
