"""Non-Maximum Suppression filter for removing overlapping matches.

Implements NMS algorithm to eliminate redundant matches that overlap
significantly with higher-scoring matches.
"""


from ..match import Match
from .match_filter import MatchFilter


class NMSFilter(MatchFilter):
    """Non-Maximum Suppression filter.

    Removes overlapping matches by calculating Intersection over Union (IoU)
    and suppressing matches that overlap above a threshold with higher-scoring
    matches.

    The algorithm:
    1. Sort matches by similarity score (descending)
    2. Keep the highest-scoring match
    3. For each remaining match:
       - Calculate IoU with all kept matches
       - If IoU > threshold with any kept match, discard it
       - Otherwise, keep it

    Example:
        >>> filter = NMSFilter(iou_threshold=0.3)
        >>> filtered = filter.filter(matches)
    """

    def __init__(self, iou_threshold: float = 0.3) -> None:
        """Initialize NMS filter.

        Args:
            iou_threshold: IoU threshold for considering matches as overlapping.
                          Range: 0.0 to 1.0
                          Lower values = more aggressive suppression
                          Higher values = keep more overlapping matches

        Raises:
            ValueError: If iou_threshold is not in range [0.0, 1.0]
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")
        self.iou_threshold = iou_threshold

    def filter(self, matches: list[Match]) -> list[Match]:
        """Apply Non-Maximum Suppression to remove overlapping matches.

        Args:
            matches: List of matches to filter

        Returns:
            Filtered list of matches with overlaps removed

        Raises:
            ValueError: If matches list contains matches without regions
        """
        if not matches:
            return matches

        # Sort by score descending
        sorted_matches = sorted(matches, key=lambda m: m.similarity, reverse=True)

        kept_matches: list[Match] = []
        for match in sorted_matches:
            # Skip matches without regions
            if match.region is None:
                continue

            # Check if this match overlaps with any kept match
            should_keep = True
            for kept in kept_matches:
                # Skip kept matches without regions
                if kept.region is None:
                    continue

                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(match, kept)

                if iou > self.iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                kept_matches.append(match)

        return kept_matches

    def _calculate_iou(self, match1: Match, match2: Match) -> float:
        """Calculate Intersection over Union between two matches.

        Args:
            match1: First match
            match2: Second match

        Returns:
            IoU value in range [0.0, 1.0]
            0.0 = no overlap, 1.0 = perfect overlap

        Raises:
            ValueError: If either match lacks a region
        """
        if match1.region is None or match2.region is None:
            raise ValueError("Cannot calculate IoU for matches without regions")

        # Calculate intersection rectangle
        x1 = max(match1.region.x, match2.region.x)
        y1 = max(match1.region.y, match2.region.y)
        x2 = min(
            match1.region.x + match1.region.width,
            match2.region.x + match2.region.width,
        )
        y2 = min(
            match1.region.y + match1.region.height,
            match2.region.y + match2.region.height,
        )

        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = match1.region.width * match1.region.height
        area2 = match2.region.width * match2.region.height
        union = area1 + area2 - intersection

        # Avoid division by zero
        if union <= 0:
            return 0.0

        return intersection / union
