"""Base color strategy abstract class.

Defines the interface for color finding strategies.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .....model.match.match import Match


class BaseColorStrategy(ABC):
    """Abstract base class for color finding strategies.

    Defines the interface that all color strategies must implement.
    Each strategy provides a different approach to finding color regions
    in a scene image.
    """

    @abstractmethod
    def find_color_regions(
        self,
        scene: np.ndarray[Any, Any],
        target_images: list[np.ndarray[Any, Any]],
        options: Any,
    ) -> list[Match]:
        """Find color regions in the scene matching the targets.

        Args:
            scene: Scene image to search
            target_images: Target images containing desired colors
            options: Color finding options

        Returns:
            List of matches representing found color regions
        """
        pass
