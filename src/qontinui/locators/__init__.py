"""Self-healing locator system for qontinui.

This module provides a multi-strategy locator system that reduces test brittleness
by using multiple methods to find UI elements and automatically recovering when
primary locators fail.

Key Components:
    - LocatorStrategy: Base interface for locator strategies
    - MultiStrategyLocator: Orchestrates multiple strategies
    - HealingManager: Manages self-healing and learning
    - HealingConfig: Configuration for healing behavior
"""

from .healing import HealingConfig, HealingManager
from .multi_strategy import MultiStrategyLocator
from .strategies import (
    ColorRegionStrategy,
    LocatorStrategy,
    RelativePositionStrategy,
    SemanticTextStrategy,
    StructuralStrategy,
    VisualPatternStrategy,
)

__all__ = [
    "LocatorStrategy",
    "VisualPatternStrategy",
    "SemanticTextStrategy",
    "RelativePositionStrategy",
    "ColorRegionStrategy",
    "StructuralStrategy",
    "MultiStrategyLocator",
    "HealingManager",
    "HealingConfig",
]
