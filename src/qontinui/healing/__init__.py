"""Self-healing system for failed element lookups.

This module provides vision-based self-healing when template matching fails.
By default, LLM healing is DISABLED for privacy and offline operation.
Users must explicitly enable LLM access.

Key Features:
- Visual search fallback (expanded threshold, multi-scale)
- Optional vision LLM healing (local via Ollama or remote via API)
- Integration with action cache for learning
- Multiple provider support (OpenAI, Anthropic, Google, Ollama)

Default Behavior (LLM Disabled):
    >>> from qontinui.healing import VisionHealer, HealingConfig
    >>>
    >>> # Default: no LLM, only visual search fallback
    >>> healer = VisionHealer()
    >>> result = healer.heal(screenshot, context, pattern)

Enable Local LLM (Ollama):
    >>> config = HealingConfig.with_ollama(model_name="llava:7b")
    >>> healer = VisionHealer(config=config)

Enable Remote LLM (requires API key):
    >>> config = HealingConfig.with_openai(api_key=os.environ["OPENAI_API_KEY"])
    >>> healer = VisionHealer(config=config)
    >>>
    >>> # Or Anthropic
    >>> config = HealingConfig.with_anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
"""

from .healing_config import HealingConfig, HealingConfigurationError
from .healing_types import (
    ElementLocation,
    HealingContext,
    HealingResult,
    HealingStrategy,
    LLMMode,
)
from .llm_client import (
    DisabledVisionClient,
    LocalVisionClient,
    RemoteVisionClient,
    VisionLLMClient,
)
from .recovery_handler import (
    ElementNotFoundError,
    HealingRecoveryHandler,
    MatchNotFoundError,
    create_healing_handler,
    enable_healing_recovery,
)
from .vision_healer import (
    VisionHealer,
    configure_healing,
    get_vision_healer,
    set_vision_healer,
)

__all__ = [
    # Main classes
    "VisionHealer",
    "HealingConfig",
    # Global access
    "get_vision_healer",
    "set_vision_healer",
    "configure_healing",
    # Types
    "HealingResult",
    "HealingContext",
    "HealingStrategy",
    "ElementLocation",
    "LLMMode",
    # LLM Clients
    "VisionLLMClient",
    "DisabledVisionClient",
    "LocalVisionClient",
    "RemoteVisionClient",
    # Errors
    "HealingConfigurationError",
    # Recovery integration
    "HealingRecoveryHandler",
    "ElementNotFoundError",
    "MatchNotFoundError",
    "create_healing_handler",
    "enable_healing_recovery",
]
