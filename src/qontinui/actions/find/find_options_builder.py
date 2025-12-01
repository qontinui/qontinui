"""FindOptions builder with full cascade logic.

This module implements the Converter Pattern to translate SearchOptions (JSON config)
to FindOptions (execution layer) with proper priority cascade for all parameters.

Cascade Priority (highest to lowest):
1. FindOptions explicit params (action-level, direct API calls)
2. SearchOptions from JSON config (action config from frontend)
3. Pattern-level overrides (image-level)
4. StateImage config (state-level from JSON)
5. Project config (QontinuiSettings)
6. Library defaults (action_defaults)

Architecture:
- CascadeContext: Holds all potential sources of configuration
- build_find_options(): Main entry point for conversion
- _cascade_*(): Individual cascade functions for each parameter
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...config.models.search import (
    MatchAdjustment,
    PatternOptions,
    PollingConfig,
    SearchOptions,
    SearchStrategy,
)
from ...model.element import Pattern, Region

if TYPE_CHECKING:
    from ...config.settings import QontinuiSettings
    from ...model.state.state_image import StateImage

from .find_options import FindOptions


@dataclass
class CascadeContext:
    """Context holding all sources for option cascade.

    Attributes:
        search_options: SearchOptions from JSON config (priority 2)
        pattern: Pattern with image-level overrides (priority 3)
        state_image: StateImage with state-level config (priority 4)
        project_config: Project-level settings (priority 5)

    Note: Priority 1 (explicit FindOptions params) is handled by caller.
    Priority 6 (library defaults) is handled by factory functions.
    """

    search_options: SearchOptions | None = None
    pattern: Pattern | None = None
    state_image: "StateImage | None" = None
    project_config: "QontinuiSettings | None" = None


def build_find_options(
    ctx: CascadeContext,
    *,
    # Allow explicit overrides (priority 1 - highest)
    explicit_similarity: float | None = None,
    explicit_timeout: float | None = None,
    explicit_search_region: Region | None = None,
    explicit_find_all: bool | None = None,
    explicit_debug: bool | None = None,
) -> FindOptions:
    """Build FindOptions applying full cascade logic.

    This is the main entry point for converting SearchOptions to FindOptions
    with proper priority handling.

    Args:
        ctx: Cascade context with all config sources
        explicit_similarity: Override similarity (priority 1)
        explicit_timeout: Override timeout (priority 1)
        explicit_search_region: Override search region (priority 1)
        explicit_find_all: Override find_all (priority 1)
        explicit_debug: Override debug flag (priority 1)

    Returns:
        FindOptions with all parameters properly cascaded

    Example:
        ```python
        # From JSON action execution
        ctx = CascadeContext(
            search_options=action.search_options,
            pattern=pattern,
            state_image=state_image,
            project_config=QontinuiSettings(),
        )
        options = build_find_options(ctx)

        # With explicit override
        options = build_find_options(ctx, explicit_similarity=0.95)
        ```
    """
    return FindOptions(
        similarity=_cascade_similarity(ctx, explicit_similarity),
        timeout=_cascade_timeout(ctx, explicit_timeout),
        search_region=_cascade_search_region(ctx, explicit_search_region),
        find_all=_cascade_find_all(ctx, explicit_find_all),
        collect_debug=_cascade_debug(ctx, explicit_debug),
    )


def _cascade_similarity(ctx: CascadeContext, explicit: float | None) -> float:
    """Cascade similarity threshold.

    Priority:
    1. explicit parameter (if provided)
    2. SearchOptions.similarity
    3. Pattern.similarity
    4. StateImage._similarity
    5. Project config (QontinuiSettings.similarity_threshold)
    6. Library default (from action_defaults)

    Args:
        ctx: Cascade context
        explicit: Explicit override (priority 1)

    Returns:
        Similarity threshold to use
    """
    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions from JSON config
    if ctx.search_options and ctx.search_options.similarity is not None:
        return ctx.search_options.similarity

    # Priority 3: Pattern-level override
    if ctx.pattern and ctx.pattern.similarity is not None:
        return ctx.pattern.similarity

    # Priority 4: StateImage threshold
    if ctx.state_image and ctx.state_image._similarity is not None:
        return ctx.state_image._similarity

    # Priority 5: Project config
    if ctx.project_config:
        return ctx.project_config.similarity_threshold

    # Priority 6: Library default (handled by FindOptions default factory)
    # Return None to trigger the factory function
    from ...config.action_defaults import get_defaults

    return get_defaults().find.default_similarity_threshold


def _cascade_timeout(ctx: CascadeContext, explicit: float | None) -> float:
    """Cascade timeout value.

    Priority:
    1. explicit parameter
    2. SearchOptions.timeout
    3. Project config (QontinuiSettings.timeout)
    4. Library default (from action_defaults)

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        Timeout in seconds
    """
    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions
    if ctx.search_options and ctx.search_options.timeout is not None:
        # Convert from milliseconds (JSON) to seconds (FindOptions)
        return ctx.search_options.timeout / 1000.0

    # Priority 3: Project config
    if ctx.project_config:
        return ctx.project_config.timeout

    # Priority 4: Library default
    from ...config.action_defaults import get_defaults

    return get_defaults().find.default_timeout


def _cascade_search_region(
    ctx: CascadeContext, explicit: Region | None
) -> Region | None:
    """Cascade search region.

    Priority:
    1. explicit parameter
    2. SearchOptions.search_regions[0] (use first if multiple)
    3. Pattern.search_region
    4. StateImage._search_region
    5. None (search full screen)

    Note: SearchOptions has search_regions (list), FindOptions has search_region (single).
    We take the first region from the list if multiple are provided.

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        Search region or None for full screen
    """
    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions (take first region if multiple)
    if ctx.search_options and ctx.search_options.search_regions:
        regions = ctx.search_options.search_regions
        if regions and len(regions) > 0:
            return regions[0]  # type: ignore[return-value]

    # Priority 3: Pattern search region
    if ctx.pattern and ctx.pattern.search_region is not None:  # type: ignore[attr-defined]
        return ctx.pattern.search_region  # type: ignore[attr-defined,no-any-return]

    # Priority 4: StateImage search region
    if ctx.state_image and ctx.state_image._search_region is not None:
        return ctx.state_image._search_region

    # Priority 5: None (full screen search)
    return None


def _cascade_find_all(ctx: CascadeContext, explicit: bool | None) -> bool:
    """Cascade find_all flag.

    Determines whether to find all matches or just the first one.

    Priority:
    1. explicit parameter
    2. SearchOptions.max_matches > 1 or min_matches > 1
    3. False (default: find first match only)

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        True to find all matches, False for first only
    """
    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: Infer from SearchOptions max/min matches
    if ctx.search_options:
        # If max_matches > 1, we need to find multiple
        if (
            ctx.search_options.max_matches is not None
            and ctx.search_options.max_matches > 1
        ):
            return True
        # If min_matches > 1, we need to find multiple
        if (
            ctx.search_options.min_matches is not None
            and ctx.search_options.min_matches > 1
        ):
            return True
        # If max_matches_to_act_on > 1, find multiple
        if (
            ctx.search_options.max_matches_to_act_on is not None
            and ctx.search_options.max_matches_to_act_on > 1
        ):
            return True

    # Priority 3: Default to single match
    return False


def _cascade_debug(ctx: CascadeContext, explicit: bool | None) -> bool:
    """Cascade debug/capture flag.

    Priority:
    1. explicit parameter
    2. SearchOptions.capture_image
    3. Project config (QontinuiSettings.debug_mode)
    4. False (default)

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        True to collect debug info, False otherwise
    """
    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions capture_image
    if ctx.search_options and ctx.search_options.capture_image is not None:
        return ctx.search_options.capture_image

    # Priority 3: Project debug mode
    if ctx.project_config and ctx.project_config.debug_mode:
        return True

    # Priority 4: Default off
    return False


# Advanced cascade functions for options not yet in FindOptions.
# These are ready to use when FindOptions is expanded to include these fields.


def _cascade_polling(
    ctx: CascadeContext, explicit: PollingConfig | None
) -> PollingConfig | None:
    """Cascade polling configuration.

    Priority:
    1. Explicit parameter
    2. SearchOptions.polling
    3. Project config polling defaults
    4. None (no polling)

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        PollingConfig or None
    """
    from ...config.models.search import PollingConfig

    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions from JSON config
    if ctx.search_options and ctx.search_options.polling is not None:
        return ctx.search_options.polling

    # Priority 3: Project config defaults
    if ctx.project_config and hasattr(ctx.project_config, "polling_interval"):
        return PollingConfig(
            interval=ctx.project_config.polling_interval,
            max_attempts=getattr(ctx.project_config, "max_polling_attempts", None),
        )

    # Priority 4: No polling by default
    return None


def _cascade_pattern_options(
    ctx: CascadeContext, explicit: PatternOptions | None
) -> PatternOptions | None:
    """Cascade pattern matching options.

    Priority:
    1. Explicit parameter
    2. SearchOptions.pattern
    3. Pattern-level pattern options
    4. Project config pattern defaults
    5. None (use library defaults)

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        PatternOptions or None
    """
    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions from JSON config
    if ctx.search_options and ctx.search_options.pattern is not None:
        return ctx.search_options.pattern

    # Priority 3: Pattern-level options
    if ctx.pattern and hasattr(ctx.pattern, "pattern_options"):
        return ctx.pattern.pattern_options  # type: ignore[attr-defined, no-any-return]

    # Priority 4: Project config defaults
    if ctx.project_config and hasattr(ctx.project_config, "pattern_options"):
        return ctx.project_config.pattern_options  # type: ignore[attr-defined, no-any-return]

    # Priority 5: None (use library defaults in FindAction)
    return None


def _cascade_match_adjustment(
    ctx: CascadeContext, explicit: MatchAdjustment | None
) -> MatchAdjustment | None:
    """Cascade match adjustment configuration.

    Priority:
    1. Explicit parameter
    2. SearchOptions.adjustment
    3. Pattern-level adjustment
    4. None (no adjustment)

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        MatchAdjustment or None
    """
    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions from JSON config
    if ctx.search_options and ctx.search_options.adjustment is not None:
        return ctx.search_options.adjustment

    # Priority 3: Pattern-level adjustment
    if ctx.pattern and hasattr(ctx.pattern, "match_adjustment"):
        return ctx.pattern.match_adjustment  # type: ignore[attr-defined, no-any-return]

    # Priority 4: No adjustment by default
    return None


def _cascade_search_strategy(
    ctx: CascadeContext, explicit: SearchStrategy | None
) -> SearchStrategy:
    """Cascade search strategy configuration.

    Priority:
    1. Explicit parameter
    2. SearchOptions.strategy
    3. Project config default strategy
    4. Library default (SINGLE_REGION)

    Args:
        ctx: Cascade context
        explicit: Explicit override

    Returns:
        SearchStrategy enum value
    """
    from ...config.models.base_types import SearchStrategy

    # Priority 1: Explicit override
    if explicit is not None:
        return explicit

    # Priority 2: SearchOptions from JSON config
    if ctx.search_options and ctx.search_options.strategy is not None:
        return ctx.search_options.strategy

    # Priority 3: Project config default
    if ctx.project_config and hasattr(ctx.project_config, "search_strategy"):
        return ctx.project_config.search_strategy  # type: ignore[attr-defined, no-any-return]

    # Priority 4: Library default
    return SearchStrategy.FIRST


# NOTE: These cascade functions are ready to use. To integrate them:
# 1. Expand FindOptions dataclass to include these fields
# 2. Update build_find_options() to call these cascade functions
# 3. Implement the corresponding logic in FindAction to use these options
