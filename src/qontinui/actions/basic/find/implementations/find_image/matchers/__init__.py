"""Template matching implementations."""

from .base_matcher import BaseMatcher
from .multiscale_matcher import MultiScaleMatcher
from .single_scale_matcher import SingleScaleMatcher

__all__ = ["BaseMatcher", "SingleScaleMatcher", "MultiScaleMatcher"]
