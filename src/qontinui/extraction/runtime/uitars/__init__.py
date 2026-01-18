"""UI-TARS integration for Qontinui.

This module provides UI-TARS (Vision-first agent) integration for:
- **Exploration**: Autonomous GUI state discovery using Thought-Action loop
- **Execution**: Runtime element grounding and action execution

## Quick Start

### Exploration (State Discovery)

```python
from qontinui.extraction.runtime.uitars import UITARSExplorer, UITARSSettings

settings = UITARSSettings(
    provider="local_transformers",
    model_size="2B",
    quantization="int4"  # For GTX 1080 (8GB)
)

explorer = UITARSExplorer(settings)
await explorer.connect(target)

trajectory = await explorer.explore(
    goal="Explore the settings menu"
)

# Convert to StateStructure
from qontinui.extraction.runtime.uitars import TrajectoryConverter
converter = TrajectoryConverter()
result = converter.convert(trajectory, output_dir=Path("./output"))
```

### Execution (Element Grounding)

```python
from qontinui.extraction.runtime.uitars import UITARSExecutor, UITARSSettings

settings = UITARSSettings(
    execution_mode="hybrid",  # Try local first, fallback to UI-TARS
    confidence_threshold=0.7
)

executor = UITARSExecutor.from_settings(settings)
result = await executor.ground_element(
    screenshot,
    "the blue Submit button"
)
```

## Configuration

Configure via environment variables with `QONTINUI_UITARS_` prefix:

```bash
# Local inference (default)
QONTINUI_UITARS_PROVIDER=local_transformers
QONTINUI_UITARS_MODEL_SIZE=2B
QONTINUI_UITARS_QUANTIZATION=int4

# Cloud inference (HuggingFace)
QONTINUI_UITARS_PROVIDER=cloud
QONTINUI_UITARS_HUGGINGFACE_ENDPOINT=https://...
QONTINUI_UITARS_HUGGINGFACE_API_TOKEN=hf_xxxxx

# Execution mode
QONTINUI_UITARS_EXECUTION_MODE=local  # local (default), uitars, hybrid
```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Check for UI-TARS dependencies
HAS_UITARS = False
HAS_TRANSFORMERS = False
HAS_BITSANDBYTES = False
HAS_MSS = False
HAS_PYAUTOGUI = False

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    import bitsandbytes  # noqa: F401

    HAS_BITSANDBYTES = True
except ImportError:
    pass

try:
    import mss  # noqa: F401

    HAS_MSS = True
except ImportError:
    pass

try:
    import pyautogui  # noqa: F401

    HAS_PYAUTOGUI = True
except ImportError:
    pass

# UI-TARS is available if core dependencies are present
HAS_UITARS = HAS_TRANSFORMERS and HAS_MSS and HAS_PYAUTOGUI

if TYPE_CHECKING:
    from .config import UITARSSettings
    from .executor import HybridGrounder, LocalGrounder, LocalGroundingResult, UITARSExecutor
    from .explorer import UITARSExplorationConfig, UITARSExplorer
    from .models import (
        ActionResult,
        ExplorationTrajectory,
        GroundingResult,
        UITARSAction,
        UITARSActionType,
        UITARSInferenceRequest,
        UITARSInferenceResult,
        UITARSStep,
        UITARSThought,
    )
    from .provider import (
        HuggingFaceEndpointProvider,
        LocalTransformersProvider,
        UITARSProviderBase,
        VLLMProvider,
        create_provider,
    )
    from .trajectory_converter import (
        ConversionResult,
        ConvertedState,
        ConvertedTransition,
        TrajectoryConverter,
    )

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Lazy import for UI-TARS components."""
    # Config
    if name == "UITARSSettings":
        from .config import UITARSSettings

        return UITARSSettings

    # Models
    if name in (
        "UITARSActionType",
        "UITARSThought",
        "UITARSAction",
        "UITARSStep",
        "ExplorationTrajectory",
        "GroundingResult",
        "ActionResult",
        "UITARSInferenceRequest",
        "UITARSInferenceResult",
    ):
        from . import models

        return getattr(models, name)

    # Provider
    if name in (
        "UITARSProviderBase",
        "HuggingFaceEndpointProvider",
        "LocalTransformersProvider",
        "VLLMProvider",
        "create_provider",
    ):
        from . import provider

        return getattr(provider, name)

    # Explorer
    if name in ("UITARSExplorer", "UITARSExplorationConfig"):
        from . import explorer

        return getattr(explorer, name)

    # Executor
    if name in (
        "UITARSExecutor",
        "HybridGrounder",
        "LocalGrounder",
        "LocalGroundingResult",
    ):
        from . import executor

        return getattr(executor, name)

    # Trajectory Converter
    if name in (
        "TrajectoryConverter",
        "ConversionResult",
        "ConvertedState",
        "ConvertedTransition",
    ):
        from . import trajectory_converter

        return getattr(trajectory_converter, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Availability flags
    "HAS_UITARS",
    "HAS_TRANSFORMERS",
    "HAS_BITSANDBYTES",
    "HAS_MSS",
    "HAS_PYAUTOGUI",
    # Config
    "UITARSSettings",
    # Models
    "UITARSActionType",
    "UITARSThought",
    "UITARSAction",
    "UITARSStep",
    "ExplorationTrajectory",
    "GroundingResult",
    "ActionResult",
    "UITARSInferenceRequest",
    "UITARSInferenceResult",
    # Provider
    "UITARSProviderBase",
    "HuggingFaceEndpointProvider",
    "LocalTransformersProvider",
    "VLLMProvider",
    "create_provider",
    # Explorer
    "UITARSExplorer",
    "UITARSExplorationConfig",
    # Executor
    "UITARSExecutor",
    "HybridGrounder",
    "LocalGrounder",
    "LocalGroundingResult",
    # Trajectory Converter
    "TrajectoryConverter",
    "ConversionResult",
    "ConvertedState",
    "ConvertedTransition",
]
