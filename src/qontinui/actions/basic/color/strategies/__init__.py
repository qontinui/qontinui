"""Color finding strategies.

Strategy pattern implementations for different color matching approaches.
"""

from .base_strategy import BaseColorStrategy
from .classification_strategy import ClassificationStrategy
from .kmeans_strategy import KMeansStrategy
from .mu_strategy import MUStrategy

__all__ = [
    "BaseColorStrategy",
    "ClassificationStrategy",
    "KMeansStrategy",
    "MUStrategy",
]
