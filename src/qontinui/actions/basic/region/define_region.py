"""Define region action - ported from Qontinui framework.

Central orchestrator for region definition operations.
"""

import logging
from dataclasses import dataclass, field

from .....actions.find import FindAction, FindOptions
from ....action_interface import ActionInterface
from ....action_result import ActionResult
from ....hal.factory import HALFactory
from ....model.element.location import Location
from ....model.element.region import Region
from ....model.match.match import Match
from ....object_collection import ObjectCollection
from .define_region_options import DefineAs, DefineRegionOptions

logger = logging.getLogger(__name__)


@dataclass
class DefineWithWindow:
    """Define region as focused window bounds."""

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Define region as active window.

        Args:
            matches: Action result to populate
            object_collections: Not used for window definition
        """
        # Get screen dimensions as window bounds (simplified)
        screen_capture = HALFactory.get_screen_capture()
        width, height = screen_capture.get_screen_size()
        region = Region(0, 0, width, height)

        match = Match(target=Location(region=region), score=1.0)
        matches.add_match(match)
        matches.defined_region = region
        matches.success = True

        logger.debug(f"Defined region as window: {region}")


@dataclass
class DefineWithMatch:
    """Define region relative to matches."""

    find_action: FindAction = field(default_factory=FindAction)

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Define region relative to matches.

        Args:
            matches: Action result to populate
            object_collections: Objects to find and define relative to
        """
        # Get configuration
        if not isinstance(matches.action_config, DefineRegionOptions):
            raise ValueError("DefineWithMatch requires DefineRegionOptions")

        options = matches.action_config

        # Find matches first using FindAction
        if object_collections:
            for obj_coll in object_collections:
                for state_image in obj_coll.state_images:
                    pattern = state_image.get_pattern()
                    if pattern:
                        find_options = FindOptions(similarity=0.8)
                        result = self.find_action.find(pattern, find_options)
                        if result.found and result.best_match:
                            # Add match to results
                            matches.add_match(result.best_match)
                            break
                if matches.matches:
                    break

        if not matches.matches:
            logger.warning("No matches found for region definition")
            return

        # Use first match as reference
        first_match = matches.matches[0]
        if not first_match.region:
            logger.warning("Match has no region for definition")
            return

        base_region = first_match.region

        # Define based on strategy
        if options.define_as == DefineAs.MATCH:
            region = Region(
                base_region.x + options.offset_x,
                base_region.y + options.offset_y,
                base_region.width + options.expand_width,
                base_region.height + options.expand_height,
            )
        elif options.define_as == DefineAs.BELOW_MATCH:
            region = Region(
                base_region.x + options.offset_x,
                base_region.y + base_region.height + options.offset_y,
                base_region.width + options.expand_width,
                100 + options.expand_height,  # Default height
            )
        elif options.define_as == DefineAs.ABOVE_MATCH:
            region = Region(
                base_region.x + options.offset_x,
                base_region.y - 100 + options.offset_y,
                base_region.width + options.expand_width,
                100 + options.expand_height,
            )
        elif options.define_as == DefineAs.LEFT_OF_MATCH:
            region = Region(
                base_region.x - 100 + options.offset_x,
                base_region.y + options.offset_y,
                100 + options.expand_width,
                base_region.height + options.expand_height,
            )
        elif options.define_as == DefineAs.RIGHT_OF_MATCH:
            region = Region(
                base_region.x + base_region.width + options.offset_x,
                base_region.y + options.offset_y,
                100 + options.expand_width,
                base_region.height + options.expand_height,
            )
        else:
            region = base_region

        matches.defined_region = region
        matches.success = True

        logger.debug(f"Defined region {options.define_as.name}: {region}")


@dataclass
class DefineInsideAnchors:
    """Define smallest region containing all anchors."""

    find_action: FindAction = field(default_factory=FindAction)

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Define smallest region containing all anchors.

        Args:
            matches: Action result to populate
            object_collections: Anchor objects
        """
        # Find all anchors using FindAction
        if object_collections:
            for obj_coll in object_collections:
                for state_image in obj_coll.state_images:
                    pattern = state_image.get_pattern()
                    if pattern:
                        find_options = FindOptions(similarity=0.8, find_all=True)
                        result = self.find_action.find(pattern, find_options)
                        if result.found:
                            for match in result.matches:
                                matches.add_match(match)

        if not matches.matches:
            logger.warning("No anchors found for region definition")
            return

        # Find bounding box
        min_x = min(m.region.x for m in matches.matches if m.region)
        min_y = min(m.region.y for m in matches.matches if m.region)
        max_x = max(m.region.x + m.region.width for m in matches.matches if m.region)
        max_y = max(m.region.y + m.region.height for m in matches.matches if m.region)

        region = Region(min_x, min_y, max_x - min_x, max_y - min_y)

        # Apply offsets and expansions
        if isinstance(matches.action_config, DefineRegionOptions):
            options = matches.action_config
            region.x += options.offset_x
            region.y += options.offset_y
            region.width += options.expand_width
            region.height += options.expand_height

        matches.defined_region = region
        matches.success = True

        logger.debug(f"Defined region inside anchors: {region}")


@dataclass
class DefineOutsideAnchors:
    """Define largest region containing all anchors."""

    find_action: FindAction = field(default_factory=FindAction)

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Define largest region containing all anchors.

        Args:
            matches: Action result to populate
            object_collections: Anchor objects
        """
        # Similar to inside anchors but with padding - find all anchors using FindAction
        if object_collections:
            for obj_coll in object_collections:
                for state_image in obj_coll.state_images:
                    pattern = state_image.get_pattern()
                    if pattern:
                        find_options = FindOptions(similarity=0.8, find_all=True)
                        result = self.find_action.find(pattern, find_options)
                        if result.found:
                            for match in result.matches:
                                matches.add_match(match)

        if not matches.matches:
            logger.warning("No anchors found for region definition")
            return

        # Find bounding box with padding
        padding = 50  # Default padding
        min_x = min(m.region.x for m in matches.matches if m.region) - padding
        min_y = min(m.region.y for m in matches.matches if m.region) - padding
        max_x = max(m.region.x + m.region.width for m in matches.matches if m.region) + padding
        max_y = max(m.region.y + m.region.height for m in matches.matches if m.region) + padding

        # Clamp to screen bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        screen_capture = HALFactory.get_screen_capture()
        width, height = screen_capture.get_screen_size()
        max_x = min(width, max_x)
        max_y = min(height, max_y)

        region = Region(min_x, min_y, max_x - min_x, max_y - min_y)

        # Apply additional offsets and expansions
        if isinstance(matches.action_config, DefineRegionOptions):
            options = matches.action_config
            region.x += options.offset_x
            region.y += options.offset_y
            region.width += options.expand_width
            region.height += options.expand_height

        matches.defined_region = region
        matches.success = True

        logger.debug(f"Defined region outside anchors: {region}")


@dataclass
class DefineIncludingMatches:
    """Define region including all matches."""

    find_action: FindAction = field(default_factory=FindAction)

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Define region including all matches.

        Args:
            matches: Action result to populate
            object_collections: Objects to find and include
        """
        # Find all objects using FindAction
        if object_collections:
            for obj_coll in object_collections:
                for state_image in obj_coll.state_images:
                    pattern = state_image.get_pattern()
                    if pattern:
                        find_options = FindOptions(similarity=0.8, find_all=True)
                        result = self.find_action.find(pattern, find_options)
                        if result.found:
                            for match in result.matches:
                                matches.add_match(match)

        if not matches.matches:
            logger.warning("No matches found for region definition")
            return

        # Find bounding box of all matches
        regions = [m.region for m in matches.matches if m.region]
        if not regions:
            logger.warning("No regions in matches")
            return

        min_x = min(r.x for r in regions)
        min_y = min(r.y for r in regions)
        max_x = max(r.x + r.width for r in regions)
        max_y = max(r.y + r.height for r in regions)

        region = Region(min_x, min_y, max_x - min_x, max_y - min_y)

        # Apply offsets and expansions
        if isinstance(matches.action_config, DefineRegionOptions):
            options = matches.action_config
            region.x += options.offset_x
            region.y += options.offset_y
            region.width += options.expand_width
            region.height += options.expand_height

        matches.defined_region = region
        matches.success = True

        logger.debug(f"Defined region including matches: {region}")


@dataclass
class DefineRegion(ActionInterface):
    """Central orchestrator for region definition.

    Port of DefineRegion from Qontinui framework class.

    Delegates to specific strategies based on DefineAs configuration.
    """

    find_action: FindAction = field(default_factory=FindAction)
    _strategies: dict[DefineAs, ActionInterface] = field(init=False)

    def __post_init__(self):
        """Initialize strategy mappings."""
        define_with_window = DefineWithWindow()
        define_with_match = DefineWithMatch(self.find_action)
        define_inside = DefineInsideAnchors(self.find_action)
        define_outside = DefineOutsideAnchors(self.find_action)
        define_including = DefineIncludingMatches(self.find_action)

        self._strategies = {
            DefineAs.FOCUSED_WINDOW: define_with_window,
            DefineAs.MATCH: define_with_match,
            DefineAs.BELOW_MATCH: define_with_match,
            DefineAs.ABOVE_MATCH: define_with_match,
            DefineAs.LEFT_OF_MATCH: define_with_match,
            DefineAs.RIGHT_OF_MATCH: define_with_match,
            DefineAs.INSIDE_ANCHORS: define_inside,
            DefineAs.OUTSIDE_ANCHORS: define_outside,
            DefineAs.INCLUDING_MATCHES: define_including,
        }

    def get_action_type(self) -> str:
        """Get action type.

        Returns:
            DEFINE action type
        """
        return "DEFINE"

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Perform region definition.

        Args:
            matches: Action result to populate
            object_collections: Objects to use for definition
        """
        # Get configuration
        if not isinstance(matches.action_config, DefineRegionOptions):
            raise ValueError("DefineRegion requires DefineRegionOptions")

        define_options = matches.action_config
        define_as = define_options.define_as

        logger.debug(f"Define as: {define_as.name}")

        # Delegate to appropriate strategy
        strategy = self._strategies.get(define_as)
        if strategy:
            strategy.perform(matches, *object_collections)
        else:
            raise ValueError(f"No strategy registered for DefineAs: {define_as}")
