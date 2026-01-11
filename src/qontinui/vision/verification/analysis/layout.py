"""Layout analysis module.

Provides analysis of UI layout patterns including:
- Element alignment detection
- Grid structure detection
- Spacing consistency analysis
- Visual hierarchy detection
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox

if TYPE_CHECKING:
    from qontinui_schemas.testing.environment import GUIEnvironment

    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)


class AlignmentAxis(str, Enum):
    """Alignment axis types."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class AlignmentEdge(str, Enum):
    """Alignment edge types."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CENTER_H = "center_horizontal"
    CENTER_V = "center_vertical"


@dataclass
class AlignmentGroup:
    """Group of elements sharing an alignment."""

    edge: AlignmentEdge
    position: int  # Position of alignment line
    elements: list[BoundingBox]
    tolerance: int = 5

    @property
    def count(self) -> int:
        """Number of aligned elements."""
        return len(self.elements)

    def is_aligned(self, bounds: BoundingBox) -> bool:
        """Check if element aligns with this group.

        Args:
            bounds: Element bounds.

        Returns:
            True if aligned.
        """
        pos = self._get_edge_position(bounds)
        return abs(pos - self.position) <= self.tolerance

    def _get_edge_position(self, bounds: BoundingBox) -> int:
        """Get position of element edge.

        Args:
            bounds: Element bounds.

        Returns:
            Edge position.
        """
        if self.edge == AlignmentEdge.LEFT:
            return int(bounds.x)
        elif self.edge == AlignmentEdge.RIGHT:
            return int(bounds.x + bounds.width)
        elif self.edge == AlignmentEdge.TOP:
            return int(bounds.y)
        elif self.edge == AlignmentEdge.BOTTOM:
            return int(bounds.y + bounds.height)
        elif self.edge == AlignmentEdge.CENTER_H:
            return int(bounds.x + bounds.width // 2)
        elif self.edge == AlignmentEdge.CENTER_V:
            return int(bounds.y + bounds.height // 2)
        return 0


@dataclass
class GridCell:
    """A cell in a detected grid."""

    row: int
    column: int
    bounds: BoundingBox
    content: BoundingBox | None = None


@dataclass
class GridAnalysis:
    """Analysis of grid structure."""

    rows: int
    columns: int
    cells: list[GridCell]
    row_heights: list[int]
    column_widths: list[int]
    row_gaps: list[int]
    column_gaps: list[int]
    bounds: BoundingBox

    @property
    def is_uniform(self) -> bool:
        """Check if grid has uniform cell sizes."""
        if not self.row_heights or not self.column_widths:
            return False

        row_variance = (
            np.std(self.row_heights) / np.mean(self.row_heights) if self.row_heights else 0
        )
        col_variance = (
            np.std(self.column_widths) / np.mean(self.column_widths) if self.column_widths else 0
        )

        return bool(row_variance < 0.1 and col_variance < 0.1)

    def get_cell(self, row: int, column: int) -> GridCell | None:
        """Get cell at position.

        Args:
            row: Row index.
            column: Column index.

        Returns:
            Grid cell or None.
        """
        for cell in self.cells:
            if cell.row == row and cell.column == column:
                return cell
        return None


@dataclass
class SpacingPattern:
    """Detected spacing pattern."""

    direction: AlignmentAxis
    values: list[int]
    dominant_value: int
    is_consistent: bool
    variance: float


@dataclass
class LayoutStructure:
    """Overall layout structure analysis."""

    alignment_groups: list[AlignmentGroup]
    grid: GridAnalysis | None
    horizontal_spacing: SpacingPattern | None
    vertical_spacing: SpacingPattern | None
    visual_hierarchy: list[list[BoundingBox]]  # Groups by visual weight
    bounds: BoundingBox


class LayoutAnalyzer:
    """Analyzes UI layout patterns.

    Detects:
    - Alignment groups (elements sharing edges)
    - Grid structures
    - Spacing patterns
    - Visual hierarchy

    Usage:
        analyzer = LayoutAnalyzer(config, environment)

        # Analyze layout of elements
        structure = analyzer.analyze_layout(elements)

        # Detect grid
        grid = analyzer.detect_grid(elements)

        # Find alignment groups
        groups = analyzer.find_alignment_groups(elements)
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize layout analyzer.

        Args:
            config: Vision configuration.
            environment: GUI environment.
        """
        self._config = config
        self._environment = environment

    def _get_alignment_tolerance(self) -> int:
        """Get alignment detection tolerance.

        Returns:
            Tolerance in pixels.
        """
        return 5

    def find_alignment_groups(
        self,
        elements: list[BoundingBox],
        edges: list[AlignmentEdge] | None = None,
        min_group_size: int = 2,
    ) -> list[AlignmentGroup]:
        """Find groups of aligned elements.

        Args:
            elements: List of element bounds.
            edges: Edges to check (default: all).
            min_group_size: Minimum elements to form a group.

        Returns:
            List of alignment groups.
        """
        if not elements:
            return []

        if edges is None:
            edges = list(AlignmentEdge)

        tolerance = self._get_alignment_tolerance()
        groups: list[AlignmentGroup] = []

        for edge in edges:
            # Get edge positions for all elements
            positions = []
            for elem in elements:
                pos = self._get_edge_position(elem, edge)
                positions.append((pos, elem))

            # Sort by position
            positions.sort(key=lambda x: x[0])

            # Find groups of aligned elements
            current_group: list[BoundingBox] = []
            current_pos = positions[0][0] if positions else 0

            for pos, elem in positions:
                if abs(pos - current_pos) <= tolerance:
                    current_group.append(elem)
                else:
                    # Save current group if large enough
                    if len(current_group) >= min_group_size:
                        avg_pos = int(
                            np.mean([self._get_edge_position(e, edge) for e in current_group])
                        )
                        groups.append(
                            AlignmentGroup(
                                edge=edge,
                                position=avg_pos,
                                elements=current_group,
                                tolerance=tolerance,
                            )
                        )

                    # Start new group
                    current_group = [elem]
                    current_pos = pos

            # Don't forget last group
            if len(current_group) >= min_group_size:
                avg_pos = int(np.mean([self._get_edge_position(e, edge) for e in current_group]))
                groups.append(
                    AlignmentGroup(
                        edge=edge,
                        position=avg_pos,
                        elements=current_group,
                        tolerance=tolerance,
                    )
                )

        return groups

    def _get_edge_position(self, bounds: BoundingBox, edge: AlignmentEdge) -> int:
        """Get position of element edge.

        Args:
            bounds: Element bounds.
            edge: Edge type.

        Returns:
            Edge position.
        """
        if edge == AlignmentEdge.LEFT:
            return int(bounds.x)
        elif edge == AlignmentEdge.RIGHT:
            return int(bounds.x + bounds.width)
        elif edge == AlignmentEdge.TOP:
            return int(bounds.y)
        elif edge == AlignmentEdge.BOTTOM:
            return int(bounds.y + bounds.height)
        elif edge == AlignmentEdge.CENTER_H:
            return int(bounds.x + bounds.width // 2)
        elif edge == AlignmentEdge.CENTER_V:
            return int(bounds.y + bounds.height // 2)
        return 0

    def detect_grid(
        self,
        elements: list[BoundingBox],
        min_rows: int = 2,
        min_cols: int = 2,
    ) -> GridAnalysis | None:
        """Detect grid structure in elements.

        Args:
            elements: List of element bounds.
            min_rows: Minimum rows for grid detection.
            min_cols: Minimum columns for grid detection.

        Returns:
            Grid analysis or None if no grid detected.
        """
        if len(elements) < min_rows * min_cols:
            return None

        tolerance = self._get_alignment_tolerance()

        # Find row alignments (elements with similar Y)
        y_positions = [(e.y, e) for e in elements]
        rows = self._cluster_positions([y for y, _ in y_positions], tolerance)

        # Find column alignments (elements with similar X)
        x_positions = [(e.x, e) for e in elements]
        cols = self._cluster_positions([x for x, _ in x_positions], tolerance)

        if len(rows) < min_rows or len(cols) < min_cols:
            return None

        # Map elements to grid positions
        cells: list[GridCell] = []
        row_heights: list[int] = []
        col_widths: list[int] = []

        for _row_idx, row_y in enumerate(sorted(rows)):
            row_elements = [e for e in elements if abs(e.y - row_y) <= tolerance]
            if row_elements:
                row_heights.append(max(e.height for e in row_elements))

        for _col_idx, col_x in enumerate(sorted(cols)):
            col_elements = [e for e in elements if abs(e.x - col_x) <= tolerance]
            if col_elements:
                col_widths.append(max(e.width for e in col_elements))

        sorted_rows = sorted(rows)
        sorted_cols = sorted(cols)

        for elem in elements:
            # Find row
            row_idx = None
            for i, row_y in enumerate(sorted_rows):
                if abs(elem.y - row_y) <= tolerance:
                    row_idx = i
                    break

            # Find column
            col_idx = None
            for i, col_x in enumerate(sorted_cols):
                if abs(elem.x - col_x) <= tolerance:
                    col_idx = i
                    break

            if row_idx is not None and col_idx is not None:
                # Calculate cell bounds
                cell_x = sorted_cols[col_idx]
                cell_y = sorted_rows[row_idx]
                cell_w = col_widths[col_idx] if col_idx < len(col_widths) else elem.width
                cell_h = row_heights[row_idx] if row_idx < len(row_heights) else elem.height

                cells.append(
                    GridCell(
                        row=row_idx,
                        column=col_idx,
                        bounds=BoundingBox(x=cell_x, y=cell_y, width=cell_w, height=cell_h),
                        content=elem,
                    )
                )

        # Calculate gaps
        row_gaps = []
        for i in range(1, len(sorted_rows)):
            gap = sorted_rows[i] - (
                sorted_rows[i - 1] + row_heights[i - 1] if i - 1 < len(row_heights) else 0
            )
            row_gaps.append(max(0, gap))

        col_gaps = []
        for i in range(1, len(sorted_cols)):
            gap = sorted_cols[i] - (
                sorted_cols[i - 1] + col_widths[i - 1] if i - 1 < len(col_widths) else 0
            )
            col_gaps.append(max(0, gap))

        # Calculate overall bounds
        if cells:
            min_x = min(c.bounds.x for c in cells)
            min_y = min(c.bounds.y for c in cells)
            max_x = max(c.bounds.x + c.bounds.width for c in cells)
            max_y = max(c.bounds.y + c.bounds.height for c in cells)
            bounds = BoundingBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)
        else:
            bounds = BoundingBox(x=0, y=0, width=0, height=0)

        return GridAnalysis(
            rows=len(sorted_rows),
            columns=len(sorted_cols),
            cells=cells,
            row_heights=row_heights,
            column_widths=col_widths,
            row_gaps=row_gaps,
            column_gaps=col_gaps,
            bounds=bounds,
        )

    def _cluster_positions(
        self,
        positions: list[int],
        tolerance: int,
    ) -> list[int]:
        """Cluster positions into groups.

        Args:
            positions: List of positions.
            tolerance: Clustering tolerance.

        Returns:
            List of cluster centers.
        """
        if not positions:
            return []

        sorted_pos = sorted(positions)
        clusters: list[list[int]] = []
        current_cluster = [sorted_pos[0]]

        for pos in sorted_pos[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                clusters.append(current_cluster)
                current_cluster = [pos]

        clusters.append(current_cluster)

        return [int(np.mean(c)) for c in clusters]

    def analyze_spacing(
        self,
        elements: list[BoundingBox],
        axis: AlignmentAxis,
    ) -> SpacingPattern | None:
        """Analyze spacing between elements.

        Args:
            elements: List of element bounds.
            axis: Axis to analyze.

        Returns:
            Spacing pattern or None.
        """
        if len(elements) < 2:
            return None

        spacings = []

        if axis == AlignmentAxis.HORIZONTAL:
            # Sort by X
            sorted_elements = sorted(elements, key=lambda e: e.x)
            for i in range(1, len(sorted_elements)):
                prev = sorted_elements[i - 1]
                curr = sorted_elements[i]
                gap = curr.x - (prev.x + prev.width)
                spacings.append(gap)
        else:
            # Sort by Y
            sorted_elements = sorted(elements, key=lambda e: e.y)
            for i in range(1, len(sorted_elements)):
                prev = sorted_elements[i - 1]
                curr = sorted_elements[i]
                gap = curr.y - (prev.y + prev.height)
                spacings.append(gap)

        if not spacings:
            return None

        # Find dominant spacing (most common)
        spacing_counts: dict[int, int] = defaultdict(int)
        tolerance = 5
        for s in spacings:
            # Round to nearest 5
            rounded = round(s / tolerance) * tolerance
            spacing_counts[rounded] += 1

        dominant = max(spacing_counts.keys(), key=lambda k: spacing_counts[k])

        variance = float(np.std(spacings)) if spacings else 0
        is_consistent = variance < 10

        return SpacingPattern(
            direction=axis,
            values=spacings,
            dominant_value=dominant,
            is_consistent=is_consistent,
            variance=variance,
        )

    def detect_visual_hierarchy(
        self,
        elements: list[BoundingBox],
        screenshot: NDArray[np.uint8] | None = None,
    ) -> list[list[BoundingBox]]:
        """Detect visual hierarchy groups.

        Groups elements by visual weight (size, position, emphasis).

        Args:
            elements: List of element bounds.
            screenshot: Optional screenshot for color analysis.

        Returns:
            List of element groups, sorted by visual weight (highest first).
        """
        if not elements:
            return []

        # Calculate visual weight for each element
        weights: list[tuple[float, BoundingBox]] = []

        for elem in elements:
            weight = 0.0

            # Size contributes to weight
            area = elem.width * elem.height
            weight += area / 10000  # Normalize

            # Top position increases weight (header elements)
            weight += (1 - elem.y / 1000) * 0.5

            # Left position slightly increases weight (LTR reading)
            weight += (1 - elem.x / 1000) * 0.2

            # Larger elements are more prominent
            weight += min(elem.width, elem.height) / 100

            weights.append((weight, elem))

        # Sort by weight descending
        weights.sort(key=lambda x: x[0], reverse=True)

        # Group into tiers
        if not weights:
            return []

        max_weight = weights[0][0]
        tiers: list[list[BoundingBox]] = [[] for _ in range(3)]

        for weight, elem in weights:
            ratio = weight / max_weight if max_weight > 0 else 0

            if ratio > 0.7:
                tiers[0].append(elem)  # Primary
            elif ratio > 0.3:
                tiers[1].append(elem)  # Secondary
            else:
                tiers[2].append(elem)  # Tertiary

        # Remove empty tiers
        return [tier for tier in tiers if tier]

    def analyze_layout(
        self,
        elements: list[BoundingBox],
        screenshot: NDArray[np.uint8] | None = None,
    ) -> LayoutStructure:
        """Perform full layout analysis.

        Args:
            elements: List of element bounds.
            screenshot: Optional screenshot for visual analysis.

        Returns:
            Complete layout structure analysis.
        """
        if not elements:
            return LayoutStructure(
                alignment_groups=[],
                grid=None,
                horizontal_spacing=None,
                vertical_spacing=None,
                visual_hierarchy=[],
                bounds=BoundingBox(x=0, y=0, width=0, height=0),
            )

        # Calculate overall bounds
        min_x = min(e.x for e in elements)
        min_y = min(e.y for e in elements)
        max_x = max(e.x + e.width for e in elements)
        max_y = max(e.y + e.height for e in elements)
        bounds = BoundingBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

        # Find alignment groups
        alignment_groups = self.find_alignment_groups(elements)

        # Detect grid
        grid = self.detect_grid(elements)

        # Analyze spacing
        h_spacing = self.analyze_spacing(elements, AlignmentAxis.HORIZONTAL)
        v_spacing = self.analyze_spacing(elements, AlignmentAxis.VERTICAL)

        # Detect visual hierarchy
        hierarchy = self.detect_visual_hierarchy(elements, screenshot)

        return LayoutStructure(
            alignment_groups=alignment_groups,
            grid=grid,
            horizontal_spacing=h_spacing,
            vertical_spacing=v_spacing,
            visual_hierarchy=hierarchy,
            bounds=bounds,
        )


__all__ = [
    "AlignmentAxis",
    "AlignmentEdge",
    "AlignmentGroup",
    "GridAnalysis",
    "GridCell",
    "LayoutAnalyzer",
    "LayoutStructure",
    "SpacingPattern",
]
