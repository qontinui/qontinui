"""Pixel-diff fallback for action-success labelling.

Used when the WSM judge is unavailable, times out, or returns a
below-threshold confidence. Mean absolute difference over the
full frame is the simplest thing that works; callers that know a
click target can crop first for better signal.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageChops

# Mean absolute channel difference per pixel above which we call the
# action "had some effect". Tuned empirically for Windows/Linux desktop
# screenshots — below ~3 is typical idle-frame noise, above ~5 is a
# real visible change somewhere on screen.
DEFAULT_PIXEL_DIFF_THRESHOLD = 5.0


@dataclass
class PixelDiffResult:
    changed: bool
    mean_abs_diff: float
    threshold: float


def pixel_diff_result(
    before_png_bytes: bytes,
    after_png_bytes: bytes,
    threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
) -> PixelDiffResult:
    """Compute the mean absolute per-channel difference between two PNGs.

    Returns a :class:`PixelDiffResult` with the raw mean and a boolean
    threshold decision. The PNGs must be the same size; otherwise the
    function returns ``changed=False`` with ``mean_abs_diff=0.0`` (since
    resizing before diffing would mask real UI changes with interpolation
    artifacts).
    """
    before = Image.open(io.BytesIO(before_png_bytes)).convert("RGB")
    after = Image.open(io.BytesIO(after_png_bytes)).convert("RGB")
    if before.size != after.size:
        return PixelDiffResult(changed=False, mean_abs_diff=0.0, threshold=threshold)

    # ImageChops.difference returns per-pixel per-channel absolute diff.
    diff = ImageChops.difference(before, after)
    # PIL Stat would be fine here; compute manually to keep the dep
    # surface minimal and avoid numpy pulling in a fresh import just for
    # a mean.
    stat_sum = 0
    w, h = diff.size
    pixel_count = w * h * 3  # 3 channels
    for band in diff.split():
        # Each band has its own histogram; sum gives total diff magnitude.
        hist = band.histogram()
        for value, count in enumerate(hist):
            stat_sum += value * count
    mean = stat_sum / pixel_count if pixel_count else 0.0
    return PixelDiffResult(
        changed=mean > threshold,
        mean_abs_diff=mean,
        threshold=threshold,
    )


def pixel_diff_verdict(
    before_png_bytes: bytes,
    after_png_bytes: bytes,
    threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
) -> tuple[bool, str]:
    """Return ``(success, reason)`` based on a pixel-diff threshold.

    Thin convenience wrapper over :func:`pixel_diff_result` that yields
    the shape callers need to populate a :class:`WSMVerdict`.
    """
    result = pixel_diff_result(before_png_bytes, after_png_bytes, threshold)
    reason = (
        f"pixel_diff mean_abs={result.mean_abs_diff:.2f} "
        f"threshold={result.threshold:.2f} changed={result.changed}"
    )
    return result.changed, reason
