"""Timing and mock execution configuration properties.

Contains simulated timings for mock mode testing, allowing tests to run
without actual GUI interactions while maintaining realistic timing.
"""

from pydantic import BaseModel, ConfigDict, Field


class MockConfig(BaseModel):
    """Mock mode timing configuration."""

    model_config = ConfigDict(validate_assignment=True)

    click_duration: float = Field(
        default=0.5, ge=0, description="Simulated click duration in seconds"
    )
    type_duration: float = Field(
        default=2.0, ge=0, description="Simulated typing duration in seconds"
    )
    find_duration: float = Field(
        default=0.3, ge=0, description="Simulated find duration in seconds"
    )
    drag_duration: float = Field(
        default=1.0, ge=0, description="Simulated drag duration in seconds"
    )
    scroll_duration: float = Field(
        default=0.5, ge=0, description="Simulated scroll duration in seconds"
    )
    wait_duration: float = Field(
        default=0.1, ge=0, description="Simulated wait duration in seconds"
    )
    vanish_duration: float = Field(
        default=1.0, ge=0, description="Simulated vanish check duration in seconds"
    )
    exists_duration: float = Field(
        default=0.3, ge=0, description="Simulated exists check duration in seconds"
    )


class TimingProperties(BaseModel):
    """Timing and mock execution configuration properties.

    Includes:
    - MockConfig: Simulated timings for mock mode testing
    """

    model_config = ConfigDict(validate_assignment=True)

    mock: MockConfig = Field(
        default_factory=MockConfig, description="Mock mode timing configuration"
    )
