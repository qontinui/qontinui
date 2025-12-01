"""Core application configuration properties.

Contains fundamental framework settings including image paths, mock mode,
startup behavior, and automation failure handling.
"""

from pydantic import BaseModel, ConfigDict, Field


class CoreConfig(BaseModel):
    """Core framework settings."""

    model_config = ConfigDict(validate_assignment=True)

    image_path: str = Field(
        default="classpath:images/", description="Path to image resources"
    )
    mock: bool = Field(
        default=False, description="Enable mock mode for testing without GUI"
    )
    headless: bool = Field(
        default=False, description="Run in headless mode without display"
    )
    sikuli_jar_path: str | None = Field(
        default=None, description="Path to SikuliX jar file"
    )
    tesseract_path: str | None = Field(
        default=None, description="Path to Tesseract executable"
    )
    image_cache_size: int = Field(
        default=100, ge=0, description="Maximum number of images to cache"
    )
    auto_wait_timeout: float = Field(
        default=3.0, ge=0, description="Default wait timeout in seconds"
    )


class StartupConfig(BaseModel):
    """Startup configuration settings."""

    model_config = ConfigDict(validate_assignment=True)

    verify_initial_states: bool = Field(
        default=False, description="Automatically verify initial states on startup"
    )
    initial_states: list[str] = Field(
        default_factory=list, description="List of state names to verify at startup"
    )
    fallback_search: bool = Field(
        default=False, description="Search all states if specified states not found"
    )
    activate_first_only: bool = Field(
        default=False, description="Activate only the first found state"
    )
    startup_delay: int = Field(
        default=0,
        ge=0,
        description="Delay in seconds before initial state verification",
    )


class AutomationConfig(BaseModel):
    """Automation failure handling configuration."""

    model_config = ConfigDict(validate_assignment=True)

    exit_on_failure: bool = Field(
        default=False, description="Exit application when automation fails"
    )
    failure_exit_code: int = Field(
        default=1, ge=0, description="Exit code when exitOnFailure is true"
    )
    throw_on_failure: bool = Field(
        default=False, description="Throw exceptions when automation fails"
    )
    log_stack_traces: bool = Field(
        default=True, description="Log stack traces for automation failures"
    )
    max_retries: int = Field(
        default=0, ge=0, description="Maximum number of automation retry attempts"
    )
    retry_delay_ms: int = Field(
        default=1000, ge=0, description="Delay in milliseconds between retry attempts"
    )
    continue_on_failure: bool = Field(
        default=False,
        description="Continue with remaining automation steps after failure",
    )
    timeout_seconds: int = Field(
        default=0,
        ge=0,
        description="Timeout in seconds for entire automation sequence (0=no timeout)",
    )
    fail_fast: bool = Field(
        default=False, description="Stop immediately on first failure without retries"
    )


class CoreProperties(BaseModel):
    """Core application configuration properties.

    Includes:
    - CoreConfig: Essential framework settings
    - StartupConfig: Startup verification behavior
    - AutomationConfig: Automation failure handling
    """

    model_config = ConfigDict(validate_assignment=True)

    core: CoreConfig = Field(
        default_factory=CoreConfig, description="Core framework settings"
    )
    startup: StartupConfig = Field(
        default_factory=StartupConfig, description="Startup configuration"
    )
    automation: AutomationConfig = Field(
        default_factory=AutomationConfig, description="Automation failure handling"
    )
