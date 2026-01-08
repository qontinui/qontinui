"""GUI Environment Discovery module.

Provides automatic visual analysis of GUI applications to discover:
- Color palettes and semantic color mappings
- Typography patterns and text sizes
- Layout regions and grid systems
- Dynamic regions (timestamps, animations)
- Visual states for UI elements (enabled/disabled, checked, etc.)
- Element patterns for template-free detection

Usage:
    from qontinui.vision.environment import GUIEnvironmentDiscovery

    # Passive discovery from screenshots
    discovery = GUIEnvironmentDiscovery()
    env = await discovery.discover_passive(screenshots)

    # Active exploration
    env = await discovery.discover_active(
        initial_state="LoginScreen",
        exploration_depth=3
    )

    # Load existing environment
    env = GUIEnvironmentDiscovery.load("app_environment.json")
"""

from qontinui.vision.environment.discovery import GUIEnvironmentDiscovery
from qontinui.vision.environment.storage import (
    export_environment_summary,
    load_environment,
    save_environment,
)

__all__ = [
    "GUIEnvironmentDiscovery",
    "export_environment_summary",
    "load_environment",
    "save_environment",
]
