"""Base classes for AI provider system.

This module defines the abstract base class for AI providers and
the data models for analysis requests and results.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisRequest:
    """Request for AI analysis.

    Attributes:
        prompt: The prompt/instruction for the AI
        working_directory: Working directory for the analysis
        results_directory: Directory containing automation results to analyze
        timeout_seconds: Maximum time to wait for analysis
        output_format: Format for output (text, json, markdown)
        context: Additional context data for the provider
    """

    prompt: str
    working_directory: str | None = None
    results_directory: str = ".automation-results/latest"
    timeout_seconds: int = 600
    output_format: str = "text"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result from AI analysis.

    Attributes:
        success: Whether the analysis completed successfully
        output: The output from the AI (if successful)
        error: Error message (if failed)
        metadata: Additional metadata about the analysis
        provider: Name of the provider that performed the analysis
    """

    success: bool
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    provider: str = ""


class AIProvider(ABC):
    """Abstract base class for AI providers.

    AI providers handle different methods of invoking AI assistants
    for analyzing automation results and fixing issues.

    Example providers:
        - ClaudeCodeProvider: Invokes Claude Code CLI
        - CustomCommandProvider: Executes user-defined commands
        - Future: OpenAIProvider, AnthropicAPIProvider, etc.
    """

    @abstractmethod
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Run analysis synchronously.

        Args:
            request: The analysis request

        Returns:
            The analysis result
        """
        pass

    @abstractmethod
    async def stream_analyze(self, request: AnalysisRequest) -> AsyncIterator[str]:
        """Stream analysis output asynchronously.

        This method yields output lines as they become available,
        enabling real-time display in the UI.

        Args:
            request: The analysis request

        Yields:
            Lines of output from the AI
        """
        # This is an async generator, must yield at least once
        yield ""  # Empty yield to satisfy type checker
        return  # Subclasses must implement

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available.

        Returns:
            True if the provider can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name.

        Returns:
            Provider name (e.g., "claude_code", "custom")
        """
        pass

    @property
    def description(self) -> str:
        """Get a human-readable description of the provider.

        Returns:
            Provider description
        """
        return f"{self.name} AI provider"
