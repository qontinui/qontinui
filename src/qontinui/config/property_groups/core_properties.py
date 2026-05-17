"""Backwards-compatibility shim - see qontinui_schemas.config.property_groups.core_properties."""

from qontinui_schemas.config.property_groups.core_properties import *  # noqa: F401,F403
from qontinui_schemas.config.property_groups.core_properties import (  # noqa: F401
    AutomationConfig,
    CoreConfig,
    CoreProperties,
    StartupConfig,
)
