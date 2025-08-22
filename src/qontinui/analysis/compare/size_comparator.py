"""Size comparator - ported from Qontinui framework.

Determines size relationships between Pattern objects.
"""

from typing import List
from ...model.element import Pattern


class SizeComparator:
    """Determines size relationships between Pattern objects for image comparison operations.
    
    Port of SizeComparator from Qontinui framework class.
    
    This utility class analyzes the dimensional relationships between two Pattern objects
    to determine if one can be completely contained within the other. This is a critical
    prerequisite for image comparison operations where the smaller image needs to be
    searched for within the larger image.
    
    The class implements a strict containment check where one pattern must be able to
    completely envelop the other in both width and height dimensions. Partial containment
    (where one dimension fits but not the other) is not supported.
    """
    
    def get_enveloped_first_or_none(self, p1: Pattern, p2: Pattern) -> List[Pattern]:
        """Determine if one Pattern can completely contain the other based on dimensions.
        
        This method performs a strict containment check where one pattern must be smaller
        than or equal to the other in both width AND height. If neither pattern can
        completely contain the other (e.g., p1 is wider but p2 is taller), an empty
        list is returned.
        
        When containment is possible, the returned list follows a specific order:
        - Index 0: The smaller (enveloped) pattern
        - Index 1: The larger (enveloping) pattern
        
        This ordering convention is relied upon by ImageComparer to set up
        the search operation correctly.
        
        Args:
            p1: The first Pattern to compare
            p2: The second Pattern to compare
            
        Returns:
            A list containing [smaller pattern, larger pattern] if one can contain the other,
            or an empty list if no complete containment is possible
        """
        patterns = []
        p1w = p1.get_b_image().get_width()
        p1h = p1.get_b_image().get_height()
        p2w = p2.get_b_image().get_width()
        p2h = p2.get_b_image().get_height()
        
        # if the width is greater and height smaller, or vice-versa, there's no fit.
        if p1w > p2w and p1h < p2h:
            return patterns
        if p1w < p2w and p1h > p2h:
            return patterns
        
        # if both are greater than or equal, that's the bigger one.
        if p1w >= p2w and p1h >= p2h:
            patterns.append(p2)
            patterns.append(p1)
            return patterns
        
        patterns.append(p1)
        patterns.append(p2)
        return patterns