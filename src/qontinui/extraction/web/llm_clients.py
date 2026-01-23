"""
LLM client adapters for natural language element selection.

This module provides unified LLM client interfaces for AI-powered web automation.
Clients are used by NaturalLanguageSelector and SelectorHealer to find elements
using natural language descriptions and to recover from broken selectors.

Supported Providers
-------------------
- **AnthropicClient**: Claude models (claude-3-5-sonnet, claude-3-opus, etc.)
- **OpenAIClient**: GPT models (gpt-4o, gpt-4-turbo, etc.)
- **LiteLLMClient**: 100+ providers via LiteLLM (Ollama, Cohere, etc.)
- **MockLLMClient**: Testing client with predefined responses

Factory Function
----------------
Use `create_llm_client()` to create clients without knowing the specific class::

    client = create_llm_client("anthropic")  # Uses ANTHROPIC_API_KEY
    client = create_llm_client("openai", model="gpt-4-turbo")
    client = create_llm_client("litellm", model="ollama/llama3.2")

Usage Examples
--------------
With NaturalLanguageSelector::

    from qontinui.extraction.web import NaturalLanguageSelector, AnthropicClient

    client = AnthropicClient()  # Uses ANTHROPIC_API_KEY env var
    selector = NaturalLanguageSelector(client)
    result = await selector.find_element("the blue submit button", elements)
    if result.found:
        print(f"Found: {result.element.text} (confidence: {result.confidence})")

With SelectorHealer::

    from qontinui.extraction.web import SelectorHealer, OpenAIClient

    client = OpenAIClient()  # Uses OPENAI_API_KEY env var
    healer = SelectorHealer(llm_client=client)
    result = await healer.heal_selector("#old-button-id", original_element, page)

Custom configuration::

    from qontinui.extraction.web import LLMConfig, AnthropicClient

    config = LLMConfig(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.1,
    )
    client = AnthropicClient(config=config)

See Also
--------
- natural_language_selector: AI-driven element selection
- selector_healer: Automatic selector repair with LLM fallback
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class LLMConfigValidationError(ValueError):
    """Raised when LLM configuration validation fails."""

    pass


@dataclass
class LLMConfig:
    """
    Configuration for LLM clients.

    Attributes
    ----------
    model : str
        Model identifier (e.g., "claude-3-5-sonnet-20241022", "gpt-4o").
    max_tokens : int
        Maximum tokens in response (default: 500).
    temperature : float
        Sampling temperature, 0.0 for deterministic (default: 0.0).
    timeout : float
        Request timeout in seconds (default: 30.0).
    extra_params : dict
        Additional provider-specific parameters.

    Raises
    ------
    LLMConfigValidationError
        If any configuration value is invalid.

    Example
    -------
    ::

        config = LLMConfig(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.1,
            extra_params={"top_p": 0.9},
        )
    """

    model: str
    max_tokens: int = 500
    temperature: float = 0.0
    timeout: float = 30.0
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate LLM configuration values.

        Raises
        ------
        LLMConfigValidationError
            If any configuration value is invalid.
        """
        if not self.model or not isinstance(self.model, str):
            raise LLMConfigValidationError("model must be a non-empty string")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise LLMConfigValidationError(
                f"max_tokens must be a positive integer, got {self.max_tokens}"
            )

        if not isinstance(self.temperature, (int, float)):
            raise LLMConfigValidationError(
                f"temperature must be a number, got {type(self.temperature).__name__}"
            )
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise LLMConfigValidationError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise LLMConfigValidationError(f"timeout must be a positive number, got {self.timeout}")

        if not isinstance(self.extra_params, dict):
            raise LLMConfigValidationError(
                f"extra_params must be a dictionary, got {type(self.extra_params).__name__}"
            )


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All LLM client implementations must inherit from this class
    and implement the `complete()` method.

    Attributes
    ----------
    config : LLMConfig
        Client configuration (model, tokens, temperature, etc.).
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or self._default_config()

    @abstractmethod
    def _default_config(self) -> LLMConfig:
        """Return default configuration for this client."""
        ...

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """
        Complete a prompt and return the response text.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM.

        Returns
        -------
        str
            The generated response text.
        """
        ...


class AnthropicClient(BaseLLMClient):
    """
    LLM client for Anthropic's Claude models.

    Requires the `anthropic` package::

        pip install anthropic

    Attributes
    ----------
    api_key : str
        Anthropic API key.

    Example
    -------
    ::

        # Using environment variable (recommended)
        client = AnthropicClient()  # Uses ANTHROPIC_API_KEY

        # Explicit API key
        client = AnthropicClient(api_key="sk-ant-...")

        # Custom model
        client = AnthropicClient(model="claude-3-opus-20240229")

        # Usage
        response = await client.complete("Hello, Claude!")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        config: LLMConfig | None = None,
    ):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Model to use. Defaults to Claude 3.5 Sonnet.
            config: Optional LLMConfig for advanced settings.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Pass api_key or set ANTHROPIC_API_KEY env var."
            )

        if config:
            super().__init__(config)
        else:
            super().__init__(LLMConfig(model=model))

        self._client: Any = None

    def _default_config(self) -> LLMConfig:
        return LLMConfig(model="claude-3-5-sonnet-20241022")

    def _get_client(self) -> Any:
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                ) from e
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def complete(self, prompt: str) -> str:
        """Complete a prompt using Claude."""
        client = self._get_client()

        try:
            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **self.config.extra_params,
            )

            # Extract text from response
            if response.content and len(response.content) > 0:
                return str(response.content[0].text)
            return ""

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIClient(BaseLLMClient):
    """
    LLM client for OpenAI's GPT models.

    Requires the `openai` package::

        pip install openai

    Attributes
    ----------
    api_key : str
        OpenAI API key.

    Example
    -------
    ::

        # Using environment variable (recommended)
        client = OpenAIClient()  # Uses OPENAI_API_KEY

        # Explicit API key
        client = OpenAIClient(api_key="sk-...")

        # Custom model
        client = OpenAIClient(model="gpt-4-turbo")

        # Usage
        response = await client.complete("Hello, GPT!")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        config: LLMConfig | None = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model to use. Defaults to GPT-4o.
            config: Optional LLMConfig for advanced settings.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Pass api_key or set OPENAI_API_KEY env var.")

        if config:
            super().__init__(config)
        else:
            super().__init__(LLMConfig(model=model))

        self._client: Any = None

    def _default_config(self) -> LLMConfig:
        return LLMConfig(model="gpt-4o")

    def _get_client(self) -> Any:
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                ) from e
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def complete(self, prompt: str) -> str:
        """Complete a prompt using GPT."""
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **self.config.extra_params,
            )

            # Extract text from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class LiteLLMClient(BaseLLMClient):
    """
    LLM client using LiteLLM for provider-agnostic access.

    Supports 100+ LLM providers through a unified interface including
    OpenAI, Anthropic, Cohere, Ollama, and many more.

    Requires the `litellm` package::

        pip install litellm

    Example
    -------
    ::

        # OpenAI models
        client = LiteLLMClient(model="gpt-4o")

        # Anthropic models
        client = LiteLLMClient(model="claude-3-5-sonnet-20241022")

        # Local Ollama models
        client = LiteLLMClient(model="ollama/llama3.2")

        # Cohere models
        client = LiteLLMClient(model="cohere/command-r-plus")

        # Usage
        response = await client.complete("Hello!")

    Notes
    -----
    API keys are automatically read from environment variables based on
    the model provider (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        config: LLMConfig | None = None,
    ):
        """
        Initialize the LiteLLM client.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            api_key: API key (optional, uses env vars by default)
            config: Optional LLMConfig for advanced settings.
        """
        self.api_key = api_key

        if config:
            super().__init__(config)
        else:
            super().__init__(LLMConfig(model=model))

    def _default_config(self) -> LLMConfig:
        return LLMConfig(model="gpt-4o")

    async def complete(self, prompt: str) -> str:
        """Complete a prompt using LiteLLM."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm package not installed. Install with: pip install litellm"
            ) from e

        try:
            response = await litellm.acompletion(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                **self.config.extra_params,
            )

            # Extract text from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""

        except Exception as e:
            logger.error(f"LiteLLM API error: {e}")
            raise


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing.

    Returns predefined responses based on simple pattern matching.
    Useful for unit tests and development without API costs.

    Attributes
    ----------
    responses : dict[str, str]
        Mapping of prompt substrings to response strings.
    call_history : list[str]
        Record of all prompts sent to the client.

    Example
    -------
    ::

        # Predefined responses
        client = MockLLMClient(responses={
            "submit": "INDEX: 3\\nCONFIDENCE: 0.95\\nREASONING: Submit button found",
            "login": "INDEX: 0\\nCONFIDENCE: 0.9\\nREASONING: Login button found",
        })

        # Check call history after tests
        result = await client.complete("find the submit button")
        assert "submit" in client.call_history[0]
    """

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Initialize the mock client.

        Args:
            responses: Dict mapping prompt substrings to responses.
                       Keys are matched case-insensitively.
        """
        super().__init__()
        self.responses = responses or {}
        self.call_history: list[str] = []

    def _default_config(self) -> LLMConfig:
        return LLMConfig(model="mock")

    async def complete(self, prompt: str) -> str:
        """Return a mock response based on prompt content."""
        self.call_history.append(prompt)

        # Check for matching patterns
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                return response

        # Default response for element selection prompts
        if "INDEX:" in prompt or "find" in prompt.lower():
            return """INDEX: 0
CONFIDENCE: 0.9
REASONING: Mock selection of first element
ALTERNATIVES: 1, 2"""

        return "Mock response"


def create_llm_client(
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Factory function to create an LLM client.

    Creates the appropriate client class based on provider name,
    with sensible defaults for each provider.

    Parameters
    ----------
    provider : str
        Provider name: "anthropic", "openai", "litellm", or "mock".
    model : str, optional
        Model name. Uses provider default if not specified:
        - anthropic: claude-3-5-sonnet-20241022
        - openai: gpt-4o
        - litellm: gpt-4o
    api_key : str, optional
        API key. Uses environment variable if not specified.
    **kwargs
        Additional arguments passed to the client constructor.

    Returns
    -------
    BaseLLMClient
        Configured LLM client instance.

    Raises
    ------
    ValueError
        If provider is not recognized.

    Example
    -------
    ::

        # Anthropic (default)
        client = create_llm_client("anthropic")

        # OpenAI with specific model
        client = create_llm_client("openai", model="gpt-4-turbo")

        # Local LLM via LiteLLM
        client = create_llm_client("litellm", model="ollama/llama3.2")

        # Mock for testing
        client = create_llm_client("mock", responses={"test": "response"})
    """
    provider = provider.lower()

    if provider == "anthropic":
        return AnthropicClient(
            api_key=api_key, model=model or "claude-3-5-sonnet-20241022", **kwargs
        )
    elif provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o", **kwargs)
    elif provider == "litellm":
        return LiteLLMClient(model=model or "gpt-4o", api_key=api_key, **kwargs)
    elif provider == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. " f"Supported: anthropic, openai, litellm, mock"
        )
