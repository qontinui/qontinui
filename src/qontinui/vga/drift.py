"""Drift detection for persisted VGA elements.

Two capabilities (plan §13 recommendation C):

1. **IoU** between predicted and last-confirmed bboxes. IoU < threshold
   (default 0.2) is flagged as structural drift.
2. **Template-anchor similarity**: crop a ~20 px patch around the
   predicted point and compare it (SSIM, pHash fallback) to the patch
   stored at last-confirmation time.

Combined confidence is ``min(vlm_confidence_proxy, template_similarity)``
clamped to [0, 1]. The caller decides fail-open vs fail-closed policy.

The interface is stable even when the optional similarity libraries are
missing — the functions degrade to returning 1.0 and log a one-shot
warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .state_machine import BBox

logger = logging.getLogger(__name__)

_ANCHOR_PATCH_HALF = 20
"""Half-size in px of the anchor patch cropped around a point."""

_WARNED_NO_SIMILARITY = False
"""Module-level flag so we only warn once per process."""


@dataclass(frozen=True)
class DriftResult:
    """Outcome of a single drift check.

    Attributes:
        is_drift: True if either the IoU is below threshold OR the
            template similarity is below ``0.5``. The caller's policy
            can ignore this flag and use the numeric components directly.
        iou: Intersection-over-union, in [0, 1].
        template_similarity: SSIM / pHash similarity in [0, 1]. ``1.0``
            when no reference patch or similarity lib is available.
        combined_confidence: ``min(iou_confidence, template_similarity)``
            clamped to [0, 1]. ``iou_confidence`` is IoU normalized so
            that the 0.2 threshold maps to 0.5.
    """

    is_drift: bool
    iou: float
    template_similarity: float
    combined_confidence: float


class DriftDetector:
    """Computes IoU + template anchor similarity.

    Instantiation is cheap; keep one per runtime invocation.
    """

    @staticmethod
    def iou(bbox_a: BBox, bbox_b: BBox) -> float:
        """Intersection-over-union between two bboxes in pixel coords.

        Returns ``0.0`` if either bbox has zero area or the two do not
        overlap.
        """
        ax0, ay0, ax1, ay1 = bbox_a.as_xyxy
        bx0, by0, bx1, by1 = bbox_b.as_xyxy

        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)

        inter_w = max(0, inter_x1 - inter_x0)
        inter_h = max(0, inter_y1 - inter_y0)
        inter = inter_w * inter_h

        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter

        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def template_anchor_similarity(
        screenshot: Any,
        bbox: BBox,
        reference_patch: Any,
    ) -> float:
        """Compare a 40 x 40 patch around ``bbox.center`` to
        ``reference_patch``.

        Resolution order:

        1. If ``reference_patch`` is None, return ``1.0`` (no anchor to
           compare against — treat as "no drift").
        2. Try ``skimage.metrics.structural_similarity`` (SSIM).
        3. Fall back to ``imagehash.phash`` Hamming distance normalized
           to [0, 1].
        4. If neither library is importable, emit a one-shot warning and
           return ``1.0``.

        Args:
            screenshot: PIL Image / numpy array / bytes — the current
                frame to sample the candidate patch from.
            bbox: The predicted bbox whose center we sample around.
            reference_patch: The stored patch from last confirmation.
                PIL Image / numpy array / None.

        Returns:
            Similarity in [0, 1]. ``1.0`` = identical.
        """
        global _WARNED_NO_SIMILARITY

        if reference_patch is None:
            return 1.0

        patch = DriftDetector._crop_anchor(screenshot, bbox)
        if patch is None:
            return 1.0

        # Try SSIM first
        try:
            import numpy as np
            from skimage.metrics import structural_similarity as ssim

            ref_arr = DriftDetector._as_gray_array(reference_patch)
            cand_arr = DriftDetector._as_gray_array(patch)
            if ref_arr is None or cand_arr is None:
                return 1.0

            # SSIM requires equal shapes; resize candidate to reference.
            if ref_arr.shape != cand_arr.shape:
                try:
                    from PIL import Image as PILImage

                    pil_cand = PILImage.fromarray(cand_arr).resize(
                        (ref_arr.shape[1], ref_arr.shape[0])
                    )
                    cand_arr = np.asarray(pil_cand)
                except Exception:
                    return 1.0

            # SSIM returns value in [-1, 1]; clamp to [0, 1].
            score = float(ssim(ref_arr, cand_arr, data_range=255))
            return (
                max(0.0, min(1.0, (score + 1.0) / 2.0))
                if score < 0
                else max(0.0, min(1.0, score))
            )
        except ImportError:
            pass
        except Exception:
            logger.debug("DriftDetector: SSIM failed", exc_info=True)

        # Fallback to pHash
        try:
            import imagehash
            from PIL import Image as PILImage

            ref_pil = (
                reference_patch
                if hasattr(reference_patch, "size")
                else PILImage.fromarray(reference_patch)
            )
            cand_pil = patch if hasattr(patch, "size") else PILImage.fromarray(patch)

            ref_hash = imagehash.phash(ref_pil)
            cand_hash = imagehash.phash(cand_pil)

            # phash is 64-bit; Hamming distance in [0, 64].
            dist = float(ref_hash - cand_hash)
            return max(0.0, 1.0 - dist / 64.0)
        except ImportError:
            pass
        except Exception:
            logger.debug("DriftDetector: pHash failed", exc_info=True)

        if not _WARNED_NO_SIMILARITY:
            logger.warning(
                "DriftDetector: no similarity library available "
                "(neither scikit-image nor imagehash). "
                "Template-anchor similarity will always return 1.0."
            )
            _WARNED_NO_SIMILARITY = True

        return 1.0

    def check(
        self,
        predicted_bbox: BBox,
        last_bbox: BBox,
        screenshot: Any = None,
        reference_patch: Any = None,
        iou_threshold: float = 0.2,
    ) -> DriftResult:
        """Run both drift checks and return a combined result.

        Args:
            predicted_bbox: bbox returned by the grounding model now.
            last_bbox: bbox stored on the VgaElement from last
                confirmation.
            screenshot: Current full-frame screenshot (optional).
            reference_patch: Patch stored at last confirmation (optional).
            iou_threshold: Below this IoU the element is flagged drifted.

        Returns:
            :class:`DriftResult`.
        """
        iou_val = self.iou(predicted_bbox, last_bbox)
        similarity = self.template_anchor_similarity(
            screenshot, predicted_bbox, reference_patch
        )

        iou_confidence = (
            min(1.0, iou_val / max(iou_threshold, 1e-6) * 0.5)
            if (iou_val < iou_threshold)
            else min(
                1.0,
                0.5 + (iou_val - iou_threshold) * 0.5 / max(1.0 - iou_threshold, 1e-6),
            )
        )

        combined = min(iou_confidence, similarity)
        is_drift = iou_val < iou_threshold or similarity < 0.5

        return DriftResult(
            is_drift=is_drift,
            iou=iou_val,
            template_similarity=similarity,
            combined_confidence=max(0.0, min(1.0, combined)),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_anchor(screenshot: Any, bbox: BBox) -> Any:
        """Crop a (2 * _ANCHOR_PATCH_HALF)^2 patch around ``bbox.center``.

        Returns None on failure.
        """
        if screenshot is None:
            return None

        try:
            from PIL import Image as PILImage

            if isinstance(screenshot, PILImage.Image):
                pil = screenshot
            else:
                import numpy as np

                if isinstance(screenshot, np.ndarray):
                    pil = PILImage.fromarray(screenshot)
                elif isinstance(screenshot, bytes):
                    import io

                    pil = PILImage.open(io.BytesIO(screenshot))
                else:
                    return None

            cx, cy = bbox.center
            half = _ANCHOR_PATCH_HALF
            left = max(0, cx - half)
            top = max(0, cy - half)
            right = min(pil.width, cx + half)
            bottom = min(pil.height, cy + half)
            if right <= left or bottom <= top:
                return None
            return pil.crop((left, top, right, bottom))
        except Exception:
            logger.debug("DriftDetector: crop failed", exc_info=True)
            return None

    @staticmethod
    def _as_gray_array(img: Any) -> Any:
        """Return a 2D grayscale numpy array or None."""
        try:
            import numpy as np
            from PIL import Image as PILImage

            if isinstance(img, PILImage.Image):
                return np.asarray(img.convert("L"))
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    return img
                if img.ndim == 3 and img.shape[2] >= 3:
                    return np.asarray(PILImage.fromarray(img).convert("L"))
            return None
        except Exception:
            return None
