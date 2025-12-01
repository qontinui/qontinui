"""Logging configuration properties.

Contains comprehensive logging settings including levels, output formats,
performance options, and enrichment features.
"""

from pydantic import BaseModel, ConfigDict, Field


class LoggingConfig(BaseModel):
    """Comprehensive logging configuration."""

    model_config = ConfigDict(validate_assignment=True)

    # Global settings
    global_level: str = Field(
        default="INFO",
        pattern="^(OFF|ERROR|WARN|INFO|DEBUG|TRACE)$",
        description="Global log level",
    )

    # Category-specific levels
    actions_level: str = Field(default="INFO", description="Log level for actions")
    transitions_level: str = Field(default="INFO", description="Log level for state transitions")
    matching_level: str = Field(default="WARN", description="Log level for pattern matching")
    performance_level: str = Field(default="INFO", description="Log level for performance metrics")
    state_level: str = Field(default="DEBUG", description="Log level for state management")

    # Output configuration
    output_format: str = Field(
        default="STRUCTURED",
        pattern="^(SIMPLE|STRUCTURED|JSON)$",
        description="Output format",
    )
    include_timestamp: bool = Field(default=True, description="Include timestamp in logs")
    include_thread: bool = Field(default=False, description="Include thread name in logs")
    include_correlation_id: bool = Field(default=True, description="Include correlation ID")

    # Performance
    async_logging: bool = Field(default=True, description="Use asynchronous logging")
    buffer_size: int = Field(default=8192, ge=1024, description="Buffer size for async logging")

    # Enrichment
    include_screenshots: bool = Field(default=False, description="Attach screenshots to logs")
    include_similarity_scores: bool = Field(default=True, description="Include similarity scores")
    include_timing_breakdown: bool = Field(default=True, description="Include timing breakdown")


class LoggingProperties(BaseModel):
    """Logging configuration properties.

    Includes:
    - LoggingConfig: Comprehensive logging configuration
    """

    model_config = ConfigDict(validate_assignment=True)

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Comprehensive logging configuration"
    )
