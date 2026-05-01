"""Image matching components for pattern finding.

This package provides various image matching implementations:
- ImageMatcher: Abstract base interface for all matchers
- TemplateMatcher: OpenCV template matching implementation
- CachedTemplateMatcher: Template matcher with caching for deterministic replay
- BatchTemplateMatcher: Multi-template batch matching with NMS (requires MTM)
"""

from .batch_template_matcher import BatchTemplateMatcher
from .cached_matcher import CachedTemplateMatcher
from .image_matcher import ImageMatcher
from .template_matcher import TemplateMatcher

__all__ = [
    "ImageMatcher",
    "TemplateMatcher",
    "CachedTemplateMatcher",
    "BatchTemplateMatcher",
]
