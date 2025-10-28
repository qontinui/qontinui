"""Base template matcher interface."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .......model.element.location import Location
from .......model.element.region import Region
from .......model.match.match import Match
from ....options.pattern_find_options import PatternFindOptions

logger = logging.getLogger(__name__)


class BaseMatcher(ABC):
    """Abstract base class for template matching strategies.

    Defines the interface for different template matching approaches
    (single-scale, multi-scale, feature-based, etc.).
    """

    @abstractmethod
    def find_matches(
        self,
        template: np.ndarray[Any, Any],
        image: np.ndarray[Any, Any],
        options: PatternFindOptions,
    ) -> list[Match]:
        """Find template matches in the given image.

        Args:
            template: Template image to search for
            image: Image to search within
            options: Pattern matching configuration

        Returns:
            List of matches found
        """
        pass

    def _create_match(self, x: int, y: int, width: int, height: int, score: float) -> Match:
        """Create a Match object from coordinates.

        Args:
            x: Left coordinate
            y: Top coordinate
            width: Template width
            height: Template height
            score: Match confidence score

        Returns:
            Match object
        """
        return Match(
            target=Location(region=Region(x, y, width, height)),
            score=score,
        )
