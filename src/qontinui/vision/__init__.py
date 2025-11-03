"""Vision module - Core data classes and utilities for visual element finding.

This module provides the fundamental building blocks for vision-based element
detection and interaction, including:

- FindOptions: Configuration for find operations
- FindResult: Results container with transformation methods
- ScreenshotCache: Efficient screenshot caching

All classes follow clean code principles with comprehensive type hints and
docstrings. The module is designed with the Single Responsibility Principle
and supports fluent configuration patterns.
"""

from .find_options import FindOptions, FindStrategy, MatchMethod, SearchType
from .find_result import FindResult
from .screenshot_cache import ScreenshotCache

__all__ = [
    # Data classes
    "FindOptions",
    "FindResult",
    "ScreenshotCache",
    # Enums
    "FindStrategy",
    "MatchMethod",
    "SearchType",
]
