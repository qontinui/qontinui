"""Target resolution for vision actions.

Handles resolving different target types (ImageTarget, CoordinatesTarget, RegionTarget)
into ActionResult objects with match locations.
"""

import logging
from typing import Any

from ..actions.action_result import ActionResult, ActionResultBuilder
from ..config import CoordinatesTarget, ImageTarget, LastFindResultTarget, RegionTarget
from ..find.match import Match as FindMatch
from ..model.element import Location, Region
from ..model.match import Match as ModelMatch

logger = logging.getLogger(__name__)


class TargetResolver:
    """Resolves different target types into ActionResult objects.

    Responsibilities:
    - Convert CoordinatesTarget to ActionResult with single match
    - Convert RegionTarget to ActionResult with single match at region center
    - Retrieve cached ActionResult for "Last Find Result"
    - Delegate ImageTarget to template matching engine
    """

    def __init__(self, context: Any, template_matcher: Any):
        """Initialize resolver with context and template matcher.

        Args:
            context: Execution context with last_action_result storage
            template_matcher: TemplateMatchEngine for ImageTarget resolution
        """
        self.context = context
        self.template_matcher = template_matcher

    def resolve(self, target_config: Any) -> ActionResult | None:
        """Resolve target configuration to ActionResult.

        Args:
            target_config: TargetConfig union (ImageTarget, CoordinatesTarget, RegionTarget, str)

        Returns:
            ActionResult with matches or None if not found
        """
        import os
        import tempfile
        from datetime import datetime

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(tempfile.gettempdir(), "qontinui_find_debug.log")
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] TARGET_RESOLVER: {msg}\n")
            except Exception:
                pass

        log_debug(f"resolve() called with target_config type: {type(target_config)}")
        log_debug(f"  target_config value: {target_config}")

        if isinstance(target_config, ImageTarget):
            log_debug("  Routing to _resolve_image_target()")
            return self._resolve_image_target(target_config)
        elif isinstance(target_config, CoordinatesTarget):
            log_debug("  Routing to _resolve_coordinates_target()")
            return self._resolve_coordinates_target(target_config)
        elif isinstance(target_config, RegionTarget):
            log_debug("  Routing to _resolve_region_target()")
            return self._resolve_region_target(target_config)
        elif isinstance(target_config, LastFindResultTarget):
            log_debug("  Routing to _resolve_last_find_result()")
            return self._resolve_last_find_result()
        else:
            log_debug("  ERROR: Unsupported target type")
            logger.error(f"Unsupported target type: {type(target_config)}")
            return None

    def _resolve_image_target(self, target: ImageTarget) -> ActionResult | None:
        """Resolve ImageTarget using registry with full options cascade.

        Uses the builder pattern to properly cascade all options from:
        1. SearchOptions (JSON config)
        2. Pattern-level overrides
        3. Project config
        4. Library defaults

        Args:
            target: ImageTarget with image_ids and search options

        Returns:
            ActionResult with found matches or None
        """
        import os
        import tempfile
        from datetime import datetime

        from qontinui import registry

        from ..actions.find import FindAction
        from ..actions.find.find_options_builder import CascadeContext, build_find_options
        from ..config.settings import QontinuiSettings
        from ..model.element import Pattern

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(tempfile.gettempdir(), "qontinui_find_debug.log")
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] TARGET_RESOLVER: {msg}\n")
            except Exception:
                pass

        log_debug("_resolve_image_target() called - USING REGISTRY WITH CASCADE")

        image_ids = target.image_ids
        log_debug(f"  image_ids: {image_ids}")
        log_debug(f"  target.search_options: {target.search_options}")

        # Get project config for cascade
        try:
            project_config = QontinuiSettings()
        except Exception:
            project_config = None
            log_debug("  Could not load project config, using defaults")

        # Extract search strategy from search_options if provided
        strategy = "FIRST"  # Default strategy
        if target.search_options and target.search_options.strategy:
            # Convert enum to string value
            strategy = (
                target.search_options.strategy.value
                if hasattr(target.search_options.strategy, "value")
                else target.search_options.strategy
            )
        log_debug(f"  Resolved strategy: {strategy}")

        logger.debug(f"Resolving ImageTarget: {len(image_ids)} image(s), strategy={strategy}")

        # Use registry to load patterns (same as IF actions)
        patterns = []
        for image_id in image_ids:
            log_debug(f"  Loading pattern for image_id: {image_id}")

            # Get image from registry
            image = registry.get_image(image_id)
            if image is None:
                log_debug(f"    Image not found in registry: {image_id}")
                continue

            metadata = registry.get_image_metadata(image_id)
            if metadata is None:
                log_debug(f"    Metadata not found for image: {image_id}")
                continue

            file_path = metadata.get("file_path")
            if not file_path:
                log_debug(f"    No file_path in metadata for image: {image_id}")
                continue

            log_debug(f"    Creating pattern from: {file_path}")
            # Create pattern WITHOUT setting similarity - let cascade handle it
            pattern = Pattern.from_file(
                img_path=file_path,
                name=metadata.get("name", image_id),
            )
            patterns.append(pattern)
            log_debug("    Pattern created successfully")

        if not patterns:
            log_debug("  No patterns loaded from registry")
            return None

        log_debug(f"  Loaded {len(patterns)} patterns, using FindAction with cascade")

        # Build FindOptions using cascade for ALL patterns
        # Each pattern gets the same cascaded options
        action = FindAction()

        if strategy == "BEST" and len(patterns) > 1:
            log_debug("  BEST strategy with multiple patterns - finding all and picking best")
            # Find all patterns and pick the one with highest score
            best_result = None
            best_score = 0.0
            best_pattern = None

            for pattern in patterns:
                # Build options with cascade for this pattern
                ctx = CascadeContext(
                    search_options=target.search_options,
                    pattern=pattern,
                    state_image=None,  # Not available in this context
                    project_config=project_config,
                )
                options = build_find_options(ctx)
                log_debug(
                    f"    Finding pattern {pattern.name} with similarity={options.similarity}"
                )

                find_result = action.find(pattern, options)
                if find_result.found and find_result.matches:
                    score = find_result.matches[0].score
                    log_debug(f"    Pattern {pattern.name}: score={score}")
                    if score > best_score:
                        best_score = score
                        best_result = find_result
                        best_pattern = pattern

            if best_result and best_pattern:
                log_debug(f"  Best match score: {best_score}")
                # Convert FindResult to ActionResult
                builder = ActionResultBuilder().with_success(True)
                for match_obj in best_result.matches:
                    # Set search_image from the Pattern that was used
                    match_obj.search_image = best_pattern.image
                    # Ensure metadata is initialized (required by FindMatch)
                    if not hasattr(match_obj, "metadata"):
                        from ..model.match import MatchMetadata

                        match_obj.metadata = MatchMetadata()
                    builder.add_match(FindMatch(match_object=match_obj))
                result = builder.build()
            else:
                log_debug("  No matches found")
                result = None
        else:
            log_debug("  Using FIRST strategy or single pattern")
            # Just find the first pattern
            pattern = patterns[0]

            # Build options with cascade for this pattern
            ctx = CascadeContext(
                search_options=target.search_options,
                pattern=pattern,
                state_image=None,  # Not available in this context
                project_config=project_config,
            )
            options = build_find_options(ctx)
            log_debug(
                f"  Using cascaded options: similarity={options.similarity}, timeout={options.timeout}"
            )

            find_result = action.find(pattern, options)

            if find_result.found and find_result.matches:
                log_debug(f"  Pattern found! {len(find_result.matches)} matches")
                # Convert FindResult to ActionResult
                builder = ActionResultBuilder().with_success(True)
                for match_obj in find_result.matches:
                    # Set search_image from the Pattern that was used
                    match_obj.search_image = pattern.image
                    # Ensure metadata is initialized (required by FindMatch)
                    if not hasattr(match_obj, "metadata"):
                        from ..model.match import MatchMetadata

                        match_obj.metadata = MatchMetadata()
                    builder.add_match(FindMatch(match_object=match_obj))
                result = builder.build()
            else:
                log_debug("  Pattern not found")
                result = None

        log_debug(f"  Final result: {result}")
        log_debug(f"  result is None: {result is None}")

        if result:
            log_debug(f"  result.success: {result.success}")
            log_debug(f"  result.matches count: {len(result.matches) if result.matches else 0}")
            self.context.update_last_action_result(result)
            log_debug("  Updated context.last_action_result")
            return result
        else:
            log_debug("  Returning None (no result)")
            return None

    def _resolve_coordinates_target(self, target: CoordinatesTarget) -> ActionResult:
        """Resolve CoordinatesTarget to ActionResult with single match.

        Args:
            target: CoordinatesTarget with x, y coordinates

        Returns:
            ActionResult with single match at coordinates
        """
        coords = target.coordinates
        logger.debug(f"Resolving CoordinatesTarget: ({coords.x}, {coords.y})")

        # Create match at exact coordinates
        # Set Location x/y to match the coordinates (center of 1x1 region)
        model_match = ModelMatch(
            score=1.0,
            target=Location(
                x=coords.x, y=coords.y, region=Region(x=coords.x, y=coords.y, width=1, height=1)
            ),
        )
        find_match = FindMatch(match_object=model_match)

        result = ActionResultBuilder().with_success(True).add_match(find_match).build()
        self.context.update_last_action_result(result)
        return result

    def _resolve_region_target(self, target: RegionTarget) -> ActionResult:
        """Resolve RegionTarget to ActionResult with match at region center.

        Args:
            target: RegionTarget with region bounds

        Returns:
            ActionResult with single match at region center
        """
        region = target.region
        center_x = region.x + region.width // 2
        center_y = region.y + region.height // 2

        logger.debug(f"Resolving RegionTarget: center=({center_x}, {center_y})")

        # Create match for full region with target at center
        model_match = ModelMatch(
            score=1.0,
            target=Location(
                x=center_x,
                y=center_y,
                region=Region(x=region.x, y=region.y, width=region.width, height=region.height),
            ),
        )
        find_match = FindMatch(match_object=model_match)

        result = ActionResultBuilder().with_success(True).add_match(find_match).build()
        self.context.update_last_action_result(result)
        return result

    def _resolve_last_find_result(self) -> ActionResult | None:
        """Retrieve cached ActionResult from previous FIND action.

        Returns:
            Cached ActionResult or None if no previous result exists
        """
        if self.context.last_action_result:
            logger.debug("Resolving Last Find Result from context")
            return self.context.last_action_result  # type: ignore[no-any-return]
        else:
            logger.error("Last Find Result requested but no previous result available")
            return None
