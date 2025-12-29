"""Target resolution for vision actions.

Handles resolving different target types (ImageTarget, CoordinatesTarget, RegionTarget)
into ActionResult objects with match locations.

Coordinate translations are handled by CoordinateService for multi-monitor support.
"""

import logging
from typing import Any

from ..actions.action_result import ActionResult, ActionResultBuilder
from ..config import (
    CoordinatesTarget,
    ImageTarget,
    LastFindResultTarget,
    RegionTarget,
    StateImageTarget,
    StateLocationTarget,
    StateRegionTarget,
)
from ..coordinates import CoordinateService
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

        from qontinui_schemas.common import utc_now

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(tempfile.gettempdir(), "qontinui_find_debug.log")
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
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
        elif isinstance(target_config, StateRegionTarget):
            log_debug("  Routing to _resolve_state_region_target()")
            return self._resolve_state_region_target(target_config)
        elif isinstance(target_config, StateLocationTarget):
            log_debug("  Routing to _resolve_state_location_target()")
            return self._resolve_state_location_target(target_config)
        elif isinstance(target_config, StateImageTarget):
            log_debug("  Routing to _resolve_state_image_target()")
            return self._resolve_state_image_target(target_config)
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

        from qontinui_schemas.common import utc_now

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
                    ts = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
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

        # Expand StateImage IDs to their pattern image IDs
        # This enables multi-pattern StateImages to be searched properly
        expanded_image_ids: list[str] = []
        # Track monitors from StateImages (first one wins) - get it BEFORE expansion
        state_image_monitors: list[int] | None = None
        for image_id in image_ids:
            # Check if this is a StateImage with multi-pattern mapping
            pattern_ids = registry.get_state_image_pattern_ids(image_id)
            if pattern_ids:
                log_debug(f"  Expanding StateImage {image_id} to patterns: {pattern_ids}")
                expanded_image_ids.extend(pattern_ids)
                # Get monitors from StateImage metadata (monitors are registered on StateImage ID)
                if state_image_monitors is None:
                    state_image_meta = registry.get_image_metadata(image_id)
                    logger.debug(
                        "[TARGET_RESOLVER] StateImage %s metadata: %s", image_id, state_image_meta
                    )
                    if state_image_meta and state_image_meta.get("monitors"):
                        state_image_monitors = state_image_meta.get("monitors")
                        log_debug(
                            f"    Got monitors from StateImage config: {state_image_monitors}"
                        )
                        logger.debug(
                            "[TARGET_RESOLVER] Using monitors from StateImage: %s",
                            state_image_monitors,
                        )
                    else:
                        logger.warning("[TARGET_RESOLVER] No monitors in StateImage metadata!")
            else:
                # Not a StateImage, use as-is
                expanded_image_ids.append(image_id)

        logger.debug(
            f"Resolving ImageTarget: {len(expanded_image_ids)} image(s) after expansion, strategy={strategy}"
        )

        # Use registry to load patterns (same as IF actions)
        patterns = []
        for image_id in expanded_image_ids:
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

            # Get monitors from metadata (set by StateImage config)
            monitors_from_meta = metadata.get("monitors")
            if state_image_monitors is None and monitors_from_meta:
                state_image_monitors = monitors_from_meta  # type: ignore[assignment]
                log_debug(f"    Got monitors from StateImage config: {state_image_monitors}")

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

        # Determine monitor to use - StateImage config takes priority over ExecutionContext
        resolved_monitor: int | None = None
        if state_image_monitors:
            # StateImage has monitors configured - use the first one
            resolved_monitor = state_image_monitors[0]
            log_debug(f"  Using monitor from StateImage config: {resolved_monitor}")
        else:
            # Fall back to ExecutionContext monitor
            resolved_monitor = getattr(self.context, "monitor_index", None)
            log_debug(f"  Using monitor from ExecutionContext: {resolved_monitor}")
            # Also log to standard logger for visibility
            logger.info(f"[TARGET_RESOLVER] ExecutionContext.monitor_index = {resolved_monitor}")
            logger.debug("[TARGET_RESOLVER] DEBUG: context.monitor_index = %s", resolved_monitor)

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
                    monitor_index=resolved_monitor,
                )
                options = build_find_options(ctx)
                log_debug(
                    f"    Finding pattern {pattern.name} with similarity={options.similarity}, monitor={options.monitor_index}"
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
                # Pass monitor_index so coordinate transformation knows which system to use
                builder = (
                    ActionResultBuilder().with_success(True).with_monitor_index(resolved_monitor)
                )
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
            log_debug(
                f"  Using FIRST strategy - trying {len(patterns)} pattern(s) until one matches"
            )
            # Try each pattern until one is found (FIRST = first match wins)
            result = None
            matched_pattern = None

            for i, pattern in enumerate(patterns):
                log_debug(f"    Trying pattern {i + 1}/{len(patterns)}: {pattern.name}")

                # Build options with cascade for this pattern (monitor already resolved above)
                ctx = CascadeContext(
                    search_options=target.search_options,
                    pattern=pattern,
                    state_image=None,  # Not available in this context
                    project_config=project_config,
                    monitor_index=resolved_monitor,
                )
                options = build_find_options(ctx)
                log_debug(
                    f"      Using cascaded options: similarity={options.similarity}, timeout={options.timeout}, monitor={options.monitor_index}"
                )

                find_result = action.find(pattern, options)

                if find_result.found and find_result.matches:
                    log_debug(
                        f"    Pattern '{pattern.name}' found! {len(find_result.matches)} matches - stopping search"
                    )
                    # Convert FindResult to ActionResult
                    # Pass monitor_index so coordinate transformation knows which system to use
                    builder = (
                        ActionResultBuilder()
                        .with_success(True)
                        .with_monitor_index(resolved_monitor)
                    )
                    for match_obj in find_result.matches:
                        # Set search_image from the Pattern that was used
                        match_obj.search_image = pattern.image
                        # Ensure metadata is initialized (required by FindMatch)
                        if not hasattr(match_obj, "metadata"):
                            from ..model.match import MatchMetadata

                            match_obj.metadata = MatchMetadata()
                        builder.add_match(FindMatch(match_object=match_obj))
                    result = builder.build()
                    matched_pattern = pattern
                    break  # First match wins - stop searching
                else:
                    log_debug(f"    Pattern '{pattern.name}' not found, trying next")

            if result is None:
                log_debug("  No patterns matched")
            else:
                log_debug(
                    f"  Match found with pattern: {matched_pattern.name if matched_pattern else 'unknown'}"
                )

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
            # Clear last_action_result so subsequent actions (e.g., CLICK) don't use stale data
            self.context.update_last_action_result(None)
            log_debug("  Cleared context.last_action_result (FIND failed)")
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
                x=coords.x,
                y=coords.y,
                region=Region(x=coords.x, y=coords.y, width=1, height=1),
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

    def _resolve_state_region_target(self, target: StateRegionTarget) -> ActionResult | None:
        """Resolve StateRegionTarget by looking up region from registry.

        Uses the region's configured monitors instead of a global default.
        Coordinates are translated using CoordinateService for multi-monitor support.

        Args:
            target: StateRegionTarget with region_id

        Returns:
            ActionResult with single match at region center, or None if not found
        """
        from qontinui import registry

        region_id = target.region_id
        logger.debug(f"Resolving StateRegionTarget: {region_id}")

        # Look up region from registry
        region_data = registry.get_region(region_id)
        if region_data is None:
            logger.error(f"StateRegion '{region_id}' not found in registry")
            return None

        x = region_data["x"]
        y = region_data["y"]
        width = region_data["width"]
        height = region_data["height"]
        monitors = region_data.get("monitors")

        # Calculate center
        center_x = x + width // 2
        center_y = y + height // 2

        logger.debug(
            f"StateRegion '{region_id}': center=({center_x}, {center_y}), monitors={monitors}"
        )

        # If monitors are configured, translate coordinates to absolute screen position
        if monitors and len(monitors) > 0:
            target_monitor = monitors[0]
            try:
                service = CoordinateService.get_instance()

                # Translate center coordinates
                center_screen = service.monitor_to_screen(center_x, center_y, target_monitor)
                center_x = center_screen.x
                center_y = center_screen.y

                # Translate region bounds
                region_screen = service.monitor_to_screen(x, y, target_monitor)
                x = region_screen.x
                y = region_screen.y

                logger.debug(f"Translated to absolute: center=({center_x}, {center_y})")
            except Exception as e:
                logger.warning(f"Failed to translate coordinates for monitor {target_monitor}: {e}")

        # Create match for full region with target at center
        model_match = ModelMatch(
            score=1.0,
            target=Location(
                x=center_x,
                y=center_y,
                region=Region(x=x, y=y, width=width, height=height),
            ),
        )
        find_match = FindMatch(match_object=model_match)

        result = ActionResultBuilder().with_success(True).add_match(find_match).build()
        self.context.update_last_action_result(result)
        return result

    def _resolve_state_location_target(self, target: StateLocationTarget) -> ActionResult | None:
        """Resolve StateLocationTarget by looking up location from registry.

        Uses the location's configured monitors instead of a global default.
        Coordinates are translated using CoordinateService for multi-monitor support.

        Args:
            target: StateLocationTarget with location_id

        Returns:
            ActionResult with single match at location, or None if not found
        """
        from qontinui import registry

        location_id = target.location_id
        logger.debug(f"Resolving StateLocationTarget: {location_id}")

        # Look up location from registry
        location_data = registry.get_location(location_id)
        if location_data is None:
            logger.error(f"StateLocation '{location_id}' not found in registry")
            return None

        x = location_data["x"]
        y = location_data["y"]
        monitors = location_data.get("monitors")

        logger.debug(f"StateLocation '{location_id}': ({x}, {y}), monitors={monitors}")

        # If monitors are configured, translate coordinates to absolute screen position
        if monitors and len(monitors) > 0:
            target_monitor = monitors[0]
            try:
                service = CoordinateService.get_instance()

                # Translate location coordinates
                screen_point = service.monitor_to_screen(x, y, target_monitor)
                x = screen_point.x
                y = screen_point.y

                logger.debug(f"Translated to absolute: ({x}, {y})")
            except Exception as e:
                logger.warning(f"Failed to translate coordinates for monitor {target_monitor}: {e}")

        # Create match at location
        model_match = ModelMatch(
            score=1.0,
            target=Location(
                x=x,
                y=y,
                region=Region(x=x, y=y, width=1, height=1),
            ),
        )
        find_match = FindMatch(match_object=model_match)

        result = ActionResultBuilder().with_success(True).add_match(find_match).build()
        self.context.update_last_action_result(result)
        return result

    def _resolve_state_image_target(self, target: StateImageTarget) -> ActionResult | None:
        """Resolve StateImageTarget by converting to ImageTarget and using same logic.

        StateImageTarget is used by navigation systems to verify state by finding
        images associated with the target state. We convert it to an ImageTarget
        and use the existing image resolution logic.

        Args:
            target: StateImageTarget with state_id and image_ids

        Returns:
            ActionResult with found matches or None if not found
        """
        import os
        import tempfile

        from qontinui_schemas.common import utc_now

        def log_debug(msg: str) -> None:
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(tempfile.gettempdir(), "qontinui_find_debug.log")
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] TARGET_RESOLVER: {msg}\n")
            except Exception:
                pass

        log_debug("_resolve_state_image_target() called")
        log_debug(f"  state_id: {target.state_id}")
        log_debug(f"  image_ids: {target.image_ids}")
        log_debug(f"  state_name: {target.state_name}")
        log_debug(f"  image_names: {target.image_names}")

        # Convert StateImageTarget to ImageTarget and use existing logic
        # The image_ids in StateImageTarget are StateImage IDs
        image_target = ImageTarget(
            type="image",
            image_ids=target.image_ids,
            search_options=None,
        )

        log_debug(f"  Converted to ImageTarget: {image_target}")

        # Use the existing image target resolution which handles StateImage expansion
        return self._resolve_image_target(image_target)
