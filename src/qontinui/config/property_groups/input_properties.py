"""Input device configuration properties.

Contains mouse action timing and behavior settings, plus SikuliX integration
for input operations.
"""

from pydantic import BaseModel, ConfigDict, Field


class MouseConfig(BaseModel):
    """Mouse action configuration."""

    model_config = ConfigDict(validate_assignment=True)

    move_delay: float = Field(default=0.5, ge=0, description="Delay for mouse movement in seconds")
    pause_before_down: float = Field(
        default=0.0, ge=0, description="Pause before mouse down in seconds"
    )
    pause_after_down: float = Field(
        default=0.0, ge=0, description="Pause after mouse down in seconds"
    )
    pause_before_up: float = Field(
        default=0.0, ge=0, description="Pause before mouse up in seconds"
    )
    pause_after_up: float = Field(default=0.0, ge=0, description="Pause after mouse up in seconds")
    click_delay: float = Field(
        default=0.0, ge=0, description="Delay between clicks in double-click"
    )
    drag_delay: float = Field(default=0.5, ge=0, description="Delay during drag operations")


class SikuliConfig(BaseModel):
    """SikuliX integration settings."""

    model_config = ConfigDict(validate_assignment=True)

    highlight: bool = Field(default=False, description="Enable SikuliX highlighting")
    highlight_duration: int = Field(default=2, ge=0, description="Highlight duration in seconds")
    auto_wait_timeout: float = Field(default=0.0, ge=0, description="Auto wait timeout in seconds")
    delay_before_mouse_down: float = Field(
        default=0.0, ge=0, description="Delay before mouse down in seconds"
    )
    delay_after_drag: float = Field(default=0.0, ge=0, description="Delay after drag in seconds")
    move_mouse_delay: float = Field(default=0.5, ge=0, description="Move mouse delay in seconds")


class InputProperties(BaseModel):
    """Input device configuration properties.

    Includes:
    - MouseConfig: Mouse action timing and behavior
    - SikuliConfig: SikuliX integration for input operations
    """

    model_config = ConfigDict(validate_assignment=True)

    mouse: MouseConfig = Field(
        default_factory=MouseConfig, description="Mouse action configuration"
    )
    sikuli: SikuliConfig = Field(
        default_factory=SikuliConfig, description="SikuliX integration settings"
    )
