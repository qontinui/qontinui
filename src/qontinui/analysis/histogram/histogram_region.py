"""Histogram region - ported from Qontinui framework.

Represents a specific region for histogram analysis.
"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class HistogramRegion:
    """Represents a specific region within images for histogram analysis.
    
    Port of HistogramRegion from Qontinui framework class.
    
    This class manages masks and histograms for a particular region
    (e.g., top-left corner, bottom-right corner, or center ellipse)
    across multiple pattern images.
    
    Each HistogramRegion maintains:
    - A list of masks defining the region boundaries for each pattern
    - Individual histograms computed for each pattern's region
    - A combined histogram aggregating all pattern histograms
    
    The combined histogram is typically computed by summing the individual
    pattern histograms, providing an overall color distribution for this
    specific region across all patterns in a Qontinui image.
    """
    
    masks: List['Mat'] = field(default_factory=list)
    """Binary masks defining this region's boundaries for each pattern image.
    Each mask has white pixels (255) in the region of interest and black pixels (0) elsewhere.
    """
    
    histograms: List['Mat'] = field(default_factory=list)
    """Individual histograms computed for this region from each pattern image.
    Each histogram represents the color distribution within the masked region.
    """
    
    histogram: Optional['Mat'] = None
    """Combined histogram aggregating all individual pattern histograms.
    This represents the overall color distribution for this region across all patterns.
    """
    
    def get_masks(self) -> List['Mat']:
        """Get the list of masks.
        
        Returns:
            List of Mat masks
        """
        return self.masks
    
    def set_masks(self, masks: List['Mat']) -> None:
        """Set the list of masks.
        
        Args:
            masks: List of Mat masks to set
        """
        self.masks = masks
    
    def get_histograms(self) -> List['Mat']:
        """Get the list of individual histograms.
        
        Returns:
            List of Mat histograms
        """
        return self.histograms
    
    def set_histograms(self, histograms: List['Mat']) -> None:
        """Set the list of individual histograms.
        
        Args:
            histograms: List of Mat histograms to set
        """
        self.histograms = histograms
    
    def get_histogram(self) -> Optional['Mat']:
        """Get the combined histogram.
        
        Returns:
            Combined Mat histogram or None
        """
        return self.histogram
    
    def set_histogram(self, histogram: 'Mat') -> None:
        """Set the combined histogram.
        
        Args:
            histogram: Combined Mat histogram to set
        """
        self.histogram = histogram


# Forward reference for OpenCV Mat
class Mat:
    """Placeholder for OpenCV Mat class."""
    pass