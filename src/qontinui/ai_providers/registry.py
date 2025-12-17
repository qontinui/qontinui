"""AI provider registry for managing and accessing providers."""

import logging

from .base import AIProvider

logger = logging.getLogger(__name__)


class AIProviderRegistry:
    """Registry for AI providers.

    This class manages the registration and retrieval of AI providers.
    Providers can be registered explicitly or auto-discovered.

    Example:
        # Register a provider
        AIProviderRegistry.register("claude_code", ClaudeCodeProvider)

        # Get a provider instance
        provider = AIProviderRegistry.get_provider("claude_code")

        # List all registered providers
        providers = AIProviderRegistry.list_providers()
    """

    _providers: dict[str, type[AIProvider]] = {}
    _instances: dict[str, AIProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[AIProvider]) -> None:
        """Register an AI provider.

        Args:
            name: Provider name (e.g., "claude_code")
            provider_class: Provider class (must inherit from AIProvider)

        Raises:
            TypeError: If provider_class doesn't inherit from AIProvider
        """
        if not issubclass(provider_class, AIProvider):
            raise TypeError(f"{provider_class} must inherit from AIProvider")

        cls._providers[name] = provider_class
        logger.debug(f"Registered AI provider: {name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister an AI provider.

        Args:
            name: Provider name to unregister
        """
        if name in cls._providers:
            del cls._providers[name]
            if name in cls._instances:
                del cls._instances[name]
            logger.debug(f"Unregistered AI provider: {name}")

    @classmethod
    def get_provider(cls, name: str) -> AIProvider:
        """Get an AI provider instance by name.

        Provider instances are cached (singleton pattern per provider type).

        Args:
            name: Provider name

        Returns:
            Provider instance

        Raises:
            KeyError: If provider is not registered
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise KeyError(f"AI provider '{name}' not found. Available providers: {available}")

        # Return cached instance if available
        if name in cls._instances:
            return cls._instances[name]

        # Create new instance and cache it
        provider_class = cls._providers[name]
        instance = provider_class()
        cls._instances[name] = instance
        logger.debug(f"Created AI provider instance: {name}")
        return instance

    @classmethod
    def get_available_provider(cls, preference: str | None = None) -> AIProvider | None:
        """Get the first available AI provider.

        Args:
            preference: Preferred provider name (checked first)

        Returns:
            First available provider instance, or None if none available
        """
        # Try preferred provider first
        if preference and preference in cls._providers:
            provider = cls.get_provider(preference)
            if provider.is_available():
                return provider

        # Try all registered providers
        for name in cls._providers:
            if name == preference:
                continue  # Already tried
            provider = cls.get_provider(name)
            if provider.is_available():
                logger.info(f"Using available AI provider: {name}")
                return provider

        logger.warning("No available AI providers found")
        return None

    @classmethod
    def list_providers(cls) -> dict[str, type[AIProvider]]:
        """List all registered providers.

        Returns:
            Dictionary of provider name -> provider class
        """
        return cls._providers.copy()

    @classmethod
    def list_available_providers(cls) -> list[str]:
        """List names of all available providers.

        Returns:
            List of provider names that are currently available
        """
        available = []
        for name in cls._providers:
            provider = cls.get_provider(name)
            if provider.is_available():
                available.append(name)
        return available

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers and instances.

        This is mainly useful for testing.
        """
        cls._providers.clear()
        cls._instances.clear()
        logger.debug("Cleared AI provider registry")
