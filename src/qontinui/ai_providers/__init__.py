"""AI provider system for Qontinui.

This module provides an extensible system for integrating AI assistants
into Qontinui automation workflows. AI providers can analyze automation
results, identify issues, and potentially fix them.

Architecture:
    - AIProvider: Abstract base class for all providers
    - AIProviderRegistry: Registry for managing providers
    - Built-in providers:
        - ClaudeCodeProvider: Claude Code CLI integration
        - CustomCommandProvider: User-defined commands
        - CustomScriptProvider: User-defined scripts

Usage:
    # Get a provider
    from qontinui.ai_providers import AIProviderRegistry, AnalysisRequest

    provider = AIProviderRegistry.get_provider("claude_code")

    # Run analysis
    request = AnalysisRequest(
        prompt="Analyze the automation results and fix any issues",
        working_directory="/path/to/project",
        results_directory=".automation-results/latest",
    )

    result = provider.analyze(request)
    if result.success:
        print(result.output)
    else:
        print(f"Error: {result.error}")

    # Stream analysis (async)
    async for line in provider.stream_analyze(request):
        print(line, end="")

Configuration:
    Providers can be configured via:
    1. Direct instantiation: provider = ClaudeCodeProvider()
    2. Registry: AIProviderRegistry.register("name", ProviderClass)
    3. Environment variables (provider-specific)

Extending:
    To add a new provider:
    1. Inherit from AIProvider
    2. Implement required methods: analyze(), stream_analyze(), is_available()
    3. Register with AIProviderRegistry.register("name", YourProvider)
"""

from .base import AIProvider, AnalysisRequest, AnalysisResult
from .claude_code import ClaudeCodeProvider
from .custom import CustomCommandProvider, CustomScriptProvider
from .registry import AIProviderRegistry

# Auto-register built-in providers
# Note: CustomCommandProvider is not registered as it requires arguments
AIProviderRegistry.register("claude_code", ClaudeCodeProvider)

__all__ = [
    # Base classes
    "AIProvider",
    "AnalysisRequest",
    "AnalysisResult",
    # Registry
    "AIProviderRegistry",
    # Providers
    "ClaudeCodeProvider",
    "CustomCommandProvider",
    "CustomScriptProvider",
]
