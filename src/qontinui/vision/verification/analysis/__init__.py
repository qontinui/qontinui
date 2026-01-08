"""Visual analysis modules for advanced verification.

Provides specialized analysis capabilities:
- Text metrics (font size, baseline, spacing)
- Layout analysis (alignment, grid detection)
- Element relationships
"""

from qontinui.vision.verification.analysis.layout import (
    AlignmentGroup,
    GridAnalysis,
    LayoutAnalyzer,
    LayoutStructure,
)
from qontinui.vision.verification.analysis.relationships import (
    Element,
    ElementGroup,
    ElementRelationship,
    RelationshipAnalyzer,
    RelationshipType,
)
from qontinui.vision.verification.analysis.text_metrics import (
    TextLine,
    TextMetrics,
    TextMetricsAnalyzer,
    TextWord,
)

__all__ = [
    # Text metrics
    "TextMetrics",
    "TextMetricsAnalyzer",
    "TextLine",
    "TextWord",
    # Layout
    "AlignmentGroup",
    "GridAnalysis",
    "LayoutAnalyzer",
    "LayoutStructure",
    # Relationships
    "Element",
    "ElementGroup",
    "ElementRelationship",
    "RelationshipAnalyzer",
    "RelationshipType",
]
