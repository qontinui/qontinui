"""Backwards-compatibility shim — see qontinui_schemas.config.property_groups.output_properties."""

from qontinui_schemas.config.property_groups.output_properties import *  # noqa: F401,F403
from qontinui_schemas.config.property_groups.output_properties import (  # noqa: F401
    DatasetConfig,
    OutputProperties,
    RecordingConfig,
    ScreenshotConfig,
)
