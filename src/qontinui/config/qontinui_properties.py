"""Backwards-compatibility shim for QontinuiProperties.

The QontinuiProperties pydantic model now lives in qontinui-schemas so it can
be shared between the runner (qontinui) and the web tier (qontinui-web)
without dragging the heavy qontinui deps into the web image.

This module re-exports the schemas version so existing
``from qontinui.config.qontinui_properties import QontinuiProperties``
imports keep working.
"""

from qontinui_schemas.config.qontinui_properties import *  # noqa: F401,F403
from qontinui_schemas.config.qontinui_properties import (  # noqa: F401
    QontinuiProperties,
)
