"""Find action module - single entry point for image finding.

This module provides the core functionality for finding images on screen with
proper configuration cascade from multiple sources.

Key Components:
    - FindAction: Main action class for finding images
    - FindOptions: Execution-layer options for find operations
    - FindResult: Result object with found matches
    - find_options_builder: Converter pattern for SearchOptions → FindOptions

Configuration Cascade:
    All find operations follow a consistent priority cascade for configuration:

    Priority (highest to lowest):
    1. FindOptions explicit params (action-level, direct API calls)
    2. SearchOptions from JSON config (action config from frontend)
    3. Pattern-level overrides (image-level)
    4. StateImage config (state-level from JSON)
    5. Project config (QontinuiSettings)
    6. Library defaults (action_defaults)

Example Usage:
    ```python
    # Simple usage - uses default cascade
    from qontinui.actions.find import FindAction
    from qontinui.model.element import Pattern

    pattern = Pattern.from_file("button.png")
    action = FindAction()
    result = action.find(pattern)  # Uses project config → library default

    # With explicit options - overrides cascade
    from qontinui.actions.find import FindOptions

    options = FindOptions(similarity=0.95)  # Explicit override
    result = action.find(pattern, options)

    # JSON action execution - uses builder for full cascade
    from qontinui.actions.find.find_options_builder import (
        CascadeContext,
        build_find_options,
    )
    from qontinui.config.settings import QontinuiSettings

    ctx = CascadeContext(
        search_options=action_config.search_options,  # From JSON
        pattern=pattern,
        project_config=QontinuiSettings(),
    )
    options = build_find_options(ctx)
    result = action.find(pattern, options)
    ```

See Also:
    - find_options_builder.py: Full cascade implementation
    - find_options.py: Execution options dataclass
    - find_action.py: Main action implementation
"""

from .find_action import FindAction
from .find_options import FindOptions
from .find_result import FindResult
from .matches import Matches

__all__ = ["FindAction", "FindOptions", "FindResult", "Matches"]
