"""Modular image finding implementation.

Provides template-based image matching with support for:
- Single-scale and multi-scale matching
- Multiple OpenCV matching methods
- Async parallel pattern search
- Screen and region capture
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ......model.match.match import Match
from .....object_collection import ObjectCollection
from ...options.pattern_find_options import PatternFindOptions
from .find_image_orchestrator import FindImageOrchestrator
from .match_method_registry import MatchMethodRegistry

logger = logging.getLogger(__name__)


@dataclass
class FindImage:
    """Template matching implementation.

    Delegates to modular FindImageOrchestrator for actual implementation.
    Provides backward-compatible interface for existing code.
    """

    _orchestrator: FindImageOrchestrator = field(default_factory=FindImageOrchestrator)

    def find(self, object_collection: ObjectCollection, options: PatternFindOptions) -> list[Match]:
        """Find images using template matching.

        Args:
            object_collection: Objects containing patterns to find
            options: Pattern find configuration

        Returns:
            List of matches found
        """
        return self._orchestrator.find(object_collection, options)

    async def find_async(
        self,
        object_collection: ObjectCollection,
        options: PatternFindOptions,
        max_concurrent: int = 15,
    ) -> list[Match]:
        """Find images asynchronously with parallel pattern matching.

        Args:
            object_collection: Objects containing patterns to find
            options: Pattern find configuration
            max_concurrent: Maximum concurrent pattern matches (ignored, uses default)

        Returns:
            List of matches found
        """
        return await self._orchestrator.find_async(object_collection, options)


@dataclass
class ImageFinder:
    """Modern image finder implementation.

    Provides hook for ML-based implementations while delegating
    to template matching by default.
    """

    # Delegate to FindImage for template matching
    _template_finder: FindImage = field(default_factory=FindImage)

    # Future ML implementation
    _ml_finder: Any | None = None

    # Configuration
    use_ml_if_available: bool = True

    def find(self, object_collection: ObjectCollection, options: PatternFindOptions) -> list[Match]:
        """Find images using best available method.

        Args:
            object_collection: Objects to find
            options: Pattern options

        Returns:
            List of matches
        """
        # Check if ML finder is available and should be used
        if self.use_ml_if_available and self._ml_finder is not None:
            logger.debug("Using ML-based image finder")
            return self._ml_finder.find(object_collection, options)

        # Fall back to template matching
        logger.debug("Using template matching finder")
        return self._template_finder.find(object_collection, options)

    async def find_async(
        self,
        object_collection: ObjectCollection,
        options: PatternFindOptions,
        max_concurrent: int = 15,
    ) -> list[Match]:
        """Find images asynchronously using best available method.

        Args:
            object_collection: Objects to find
            options: Pattern options
            max_concurrent: Maximum concurrent pattern matches

        Returns:
            List of matches
        """
        # Check if ML finder is available and has async support
        if self.use_ml_if_available and self._ml_finder is not None:
            if hasattr(self._ml_finder, "find_async"):
                logger.debug("Using ML-based image finder (async)")
                return await self._ml_finder.find_async(object_collection, options)
            else:
                # Fallback to sync ML finder in thread pool
                import asyncio

                logger.debug("Using ML-based image finder (sync in thread pool)")
                return await asyncio.to_thread(self._ml_finder.find, object_collection, options)

        # Use template matching (async)
        logger.debug("Using template matching finder (async)")
        return await self._template_finder.find_async(object_collection, options, max_concurrent)

    def set_ml_finder(self, ml_finder: Any):
        """Set ML-based finder implementation.

        Args:
            ml_finder: ML finder implementation
        """
        self._ml_finder = ml_finder
        logger.info("ML finder registered")


__all__ = [
    "FindImage",
    "ImageFinder",
    "FindImageOrchestrator",
    "MatchMethodRegistry",
]
