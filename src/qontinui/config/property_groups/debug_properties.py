"""Debug and testing configuration properties.

Contains settings for test execution, GUI access verification, and console
action reporting for development and debugging.
"""

from pydantic import BaseModel, ConfigDict, Field


class TestingConfig(BaseModel):
    """Test execution settings."""

    model_config = ConfigDict(validate_assignment=True)

    timeout_multiplier: float = Field(
        default=2.0, ge=1.0, description="Multiply timeouts during testing"
    )
    retry_failed: bool = Field(
        default=True, description="Automatically retry failed tests"
    )
    max_retries: int = Field(default=3, ge=0, description="Maximum test retry attempts")
    screenshot_on_failure: bool = Field(
        default=True, description="Capture screenshot on test failure"
    )
    verbose_logging: bool = Field(
        default=True, description="Enable verbose logging during tests"
    )
    parallel_execution: bool = Field(
        default=False, description="Enable parallel test execution"
    )
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducible tests"
    )
    iteration: int = Field(default=1, ge=1, description="Current test iteration")
    send_logs: bool = Field(default=True, description="Send logs to external systems")


class GuiAccessConfig(BaseModel):
    """GUI access verification configuration."""

    model_config = ConfigDict(validate_assignment=True)

    report_problems: bool = Field(
        default=True, description="Report GUI access problems"
    )
    verbose_errors: bool = Field(default=True, description="Show verbose error details")
    suggest_solutions: bool = Field(
        default=True, description="Suggest solutions for detected problems"
    )
    check_on_startup: bool = Field(
        default=True, description="Check GUI access on startup"
    )
    continue_on_error: bool = Field(
        default=True, description="Continue execution despite GUI problems"
    )
    platform_specific_advice: bool = Field(
        default=True, description="Include platform-specific advice"
    )


class ConsoleActionConfig(BaseModel):
    """Console action reporting configuration."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(default=True, description="Enable console action reporting")
    level: str = Field(
        default="NORMAL",
        pattern="^(QUIET|NORMAL|VERBOSE)$",
        description="Verbosity level",
    )
    show_timing: bool = Field(
        default=True, description="Show timing information for actions"
    )
    use_colors: bool = Field(default=True, description="Use colored output (ANSI)")
    use_icons: bool = Field(default=True, description="Use unicode icons in output")

    # Performance thresholds
    performance_warn_threshold: int = Field(
        default=1000, ge=0, description="Warning threshold (milliseconds)"
    )
    performance_error_threshold: int = Field(
        default=5000, ge=0, description="Error threshold (milliseconds)"
    )

    # Action reporting settings
    console_actions: bool = Field(
        default=True, description="Enable console action output"
    )
    report_individual_actions: bool = Field(
        default=True, description="Report each individual action as it executes"
    )


class DebugProperties(BaseModel):
    """Debug and testing configuration properties.

    Includes:
    - TestingConfig: Test execution settings
    - GuiAccessConfig: GUI access verification
    - ConsoleActionConfig: Console action reporting
    """

    model_config = ConfigDict(validate_assignment=True)

    testing: TestingConfig = Field(
        default_factory=TestingConfig, description="Test execution settings"
    )
    gui_access: GuiAccessConfig = Field(
        default_factory=GuiAccessConfig, description="GUI access verification"
    )
    console: ConsoleActionConfig = Field(
        default_factory=ConsoleActionConfig, description="Console action reporting"
    )
