"""Backwards-compatibility shim — see qontinui_schemas.config.property_groups.debug_properties."""

from qontinui_schemas.config.property_groups.debug_properties import *  # noqa: F401,F403
from qontinui_schemas.config.property_groups.debug_properties import (  # noqa: F401
    ConsoleActionConfig,
    DebugProperties,
    GuiAccessConfig,
    TestingConfig,
)
