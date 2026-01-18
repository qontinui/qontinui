"""Configuration for self-healing behavior.

Provides configuration options for LLM mode, model selection,
and healing behavior.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .healing_types import LLMMode

if TYPE_CHECKING:
    from .llm_client import VisionLLMClient


class HealingConfigurationError(Exception):
    """Error in healing configuration."""

    pass


@dataclass
class HealingConfig:
    """Configuration for self-healing behavior.

    Controls whether and how LLM is used for element healing.
    Default is DISABLED - no remote access, fully offline.

    Attributes:
        llm_mode: How LLM is accessed (disabled, local, remote).
        local_model_name: Ollama model name for LOCAL mode.
        local_base_url: Base URL for Ollama API.
        remote_provider: Provider for REMOTE mode (openai, anthropic).
        remote_api_key: API key for REMOTE mode. Required if using remote.
        remote_model: Model name for REMOTE mode.
        max_heal_attempts: Maximum healing attempts per failure.
        heal_timeout_seconds: Timeout for healing operations.
        cache_healed_locations: Whether to cache healed locations.
    """

    # LLM mode - default is DISABLED (no remote calls, fully offline)
    llm_mode: LLMMode = LLMMode.DISABLED

    # Local model settings (only used if llm_mode == LOCAL)
    local_model_name: str = "llava:7b"
    """Ollama model name. Popular options: llava:7b, llava:13b, bakllava"""

    local_base_url: str = "http://localhost:11434"
    """Base URL for Ollama API."""

    # Remote API settings (only used if llm_mode == REMOTE)
    # User must explicitly provide these
    remote_provider: str = "openai"
    """Remote provider: 'openai', 'anthropic', 'google'."""

    remote_api_key: str | None = None
    """API key. REQUIRED for remote mode. Never logged or stored."""

    remote_model: str | None = None
    """Model override. Default depends on provider."""

    remote_base_url: str | None = None
    """Optional base URL override for remote API."""

    # Healing behavior
    max_heal_attempts: int = 2
    """Maximum healing attempts before giving up."""

    heal_timeout_seconds: float = 30.0
    """Timeout for healing operations."""

    cache_healed_locations: bool = True
    """Whether to cache healed locations for future use."""

    # Internal - created clients
    _client: "VisionLLMClient | None" = field(default=None, repr=False)

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            HealingConfigurationError: If configuration is invalid.
        """
        if self.llm_mode == LLMMode.REMOTE:
            if not self.remote_api_key:
                raise HealingConfigurationError(
                    "remote_api_key is required when llm_mode is REMOTE. "
                    "Remote LLM access must be explicitly enabled with an API key."
                )

            valid_providers = {"openai", "anthropic", "google"}
            if self.remote_provider not in valid_providers:
                raise HealingConfigurationError(
                    f"remote_provider must be one of {valid_providers}, "
                    f"got '{self.remote_provider}'"
                )

        if self.max_heal_attempts < 1:
            raise HealingConfigurationError("max_heal_attempts must be at least 1")

        if self.heal_timeout_seconds <= 0:
            raise HealingConfigurationError("heal_timeout_seconds must be positive")

    def create_client(self) -> "VisionLLMClient":
        """Create LLM client based on configuration.

        Returns:
            Configured VisionLLMClient instance.

        Raises:
            HealingConfigurationError: If configuration is invalid.
        """
        # Import here to avoid circular imports
        from .llm_client import (
            DisabledVisionClient,
            LocalVisionClient,
            RemoteVisionClient,
        )

        self.validate()

        if self.llm_mode == LLMMode.DISABLED:
            return DisabledVisionClient()

        elif self.llm_mode == LLMMode.LOCAL:
            return LocalVisionClient(
                model_name=self.local_model_name,
                base_url=self.local_base_url,
                timeout_seconds=self.heal_timeout_seconds,
            )

        elif self.llm_mode == LLMMode.REMOTE:
            # API key is validated in validate() - guaranteed non-None here
            assert self.remote_api_key is not None
            return RemoteVisionClient(
                provider=self.remote_provider,
                api_key=self.remote_api_key,
                model=self.remote_model,
                base_url=self.remote_base_url,
                timeout_seconds=self.heal_timeout_seconds,
            )

        else:
            raise HealingConfigurationError(f"Unknown LLM mode: {self.llm_mode}")

    def get_client(self) -> "VisionLLMClient":
        """Get or create the LLM client.

        Caches the client instance for reuse.

        Returns:
            VisionLLMClient instance.
        """
        if self._client is None:
            self._client = self.create_client()
        return self._client

    @classmethod
    def disabled(cls) -> "HealingConfig":
        """Create a disabled configuration (default).

        Returns:
            HealingConfig with LLM disabled.
        """
        return cls(llm_mode=LLMMode.DISABLED)

    @classmethod
    def with_ollama(
        cls,
        model_name: str = "llava:7b",
        base_url: str = "http://localhost:11434",
    ) -> "HealingConfig":
        """Create configuration for local Ollama model.

        Args:
            model_name: Ollama model name.
            base_url: Ollama API URL.

        Returns:
            HealingConfig for local model.
        """
        return cls(
            llm_mode=LLMMode.LOCAL,
            local_model_name=model_name,
            local_base_url=base_url,
        )

    @classmethod
    def with_openai(
        cls,
        api_key: str,
        model: str = "gpt-4o",
    ) -> "HealingConfig":
        """Create configuration for OpenAI API.

        Args:
            api_key: OpenAI API key.
            model: Model name.

        Returns:
            HealingConfig for OpenAI.
        """
        return cls(
            llm_mode=LLMMode.REMOTE,
            remote_provider="openai",
            remote_api_key=api_key,
            remote_model=model,
        )

    @classmethod
    def with_anthropic(
        cls,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ) -> "HealingConfig":
        """Create configuration for Anthropic API.

        Args:
            api_key: Anthropic API key.
            model: Model name.

        Returns:
            HealingConfig for Anthropic.
        """
        return cls(
            llm_mode=LLMMode.REMOTE,
            remote_provider="anthropic",
            remote_api_key=api_key,
            remote_model=model,
        )
