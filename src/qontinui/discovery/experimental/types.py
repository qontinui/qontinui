"""
EXPERIMENTAL: Type definitions for experimental detectors.

WARNING: This module is experimental and may change without notice.
Do not use in production code.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BBox:
    """Bounding box representation for experimental detectors.

    EXPERIMENTAL: This class is used by experimental detection algorithms
    and may have a different interface than the main qontinui BBox types.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    confidence: float = 1.0

    @property
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate the center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self) -> Dict:
        """Convert bounding box to dictionary representation."""
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "label": self.label,
            "confidence": self.confidence,
            "area": self.area,
        }


@dataclass
class ScreenshotInfo:
    """Information about a screenshot for multi-screenshot detection.

    EXPERIMENTAL: Used by consistency detector and other multi-screenshot
    detection algorithms.
    """

    screenshot_id: int
    path: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MultiScreenshotDataset:
    """Dataset containing multiple screenshots for analysis.

    EXPERIMENTAL: Used by detectors that analyze multiple screenshots
    to find consistent elements.
    """

    screenshots: List[ScreenshotInfo]
    name: str = ""

    def __len__(self) -> int:
        return len(self.screenshots)
