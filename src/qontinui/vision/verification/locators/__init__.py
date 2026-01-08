"""Vision locator implementations.

Provides various locator types for targeting elements on screen:
- Image locators (template matching)
- Text locators (OCR)
- Region locators (coordinate-based)
- Environment locators (using discovered data)
- Semantic locators (ML-based)
"""

from qontinui.vision.verification.locators.base import BaseLocator, LocatorMatch
from qontinui.vision.verification.locators.environment import EnvironmentLocator
from qontinui.vision.verification.locators.image import ImageLocator
from qontinui.vision.verification.locators.region import RegionLocator
from qontinui.vision.verification.locators.text import TextLocator

__all__ = [
    "BaseLocator",
    "LocatorMatch",
    "ImageLocator",
    "TextLocator",
    "RegionLocator",
    "EnvironmentLocator",
]
