"""Lazy access to the optional ML dependencies (the [ml] extra).

The default ``pip install qontinui`` is torch-free; torch, torchvision,
transformers, and friends are only present when installed with
``pip install qontinui[ml]``. Modules that need them import lazily via this
helper so package import stays cheap and works without the extra, while
instantiation fails loudly with an actionable message.
"""

from types import ModuleType
from typing import cast


def require_torch(feature: str) -> ModuleType:
    """Import and return torch, or raise naming the missing [ml] extra.

    Args:
        feature: Human-readable name of the feature needing torch, used in
            the error message (e.g. "CLIPEmbedder").

    Returns:
        The imported torch module.

    Raises:
        ImportError: When torch is not installed, with install instructions.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            f"{feature} requires the ML dependencies (torch et al.), which are "
            "not installed. Install them with: pip install qontinui[ml]"
        ) from e
    return cast(ModuleType, torch)
