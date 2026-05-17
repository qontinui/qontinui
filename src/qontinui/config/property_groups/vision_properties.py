"""Backwards-compatibility shim — see qontinui_schemas.config.property_groups.vision_properties."""

from qontinui_schemas.config.property_groups.vision_properties import *  # noqa: F401,F403
from qontinui_schemas.config.property_groups.vision_properties import (  # noqa: F401
    AnalysisConfig,
    AutoScalingConfig,
    ImageDebugConfig,
    VisionProperties,
)
