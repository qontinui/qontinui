"""
Keyboard action configuration models.

This module provides configuration models for all keyboard-related actions
including typing, key presses, and hotkey combinations.
"""

from pydantic import BaseModel, Field

from .targets import TargetConfig


class TextSource(BaseModel):
    """Text source from state string."""

    state_id: str = Field(alias="stateId")
    string_ids: list[str] = Field(alias="stringIds")
    use_all: bool | None = Field(None, alias="useAll")

    model_config = {"populate_by_name": True}


class TypeActionConfig(BaseModel):
    """TYPE action configuration."""

    text: str | None = None
    text_source: TextSource | None = Field(None, alias="textSource")
    type_delay: int | None = Field(None, alias="typeDelay")
    modifiers: list[str] | None = None
    click_target: TargetConfig | None = Field(None, alias="clickTarget")
    clear_before: bool | None = Field(None, alias="clearBefore")
    press_enter: bool | None = Field(None, alias="pressEnter")

    model_config = {"populate_by_name": True}


class KeyPressActionConfig(BaseModel):
    """KEY_PRESS action configuration."""

    keys: list[str]
    modifiers: list[str] | None = None
    hold_duration: int | None = Field(None, alias="holdDuration")
    pause_between_keys: int | None = Field(None, alias="pauseBetweenKeys")

    model_config = {"populate_by_name": True}


class KeyDownActionConfig(BaseModel):
    """KEY_DOWN action configuration."""

    keys: list[str]
    modifiers: list[str] | None = None


class KeyUpActionConfig(BaseModel):
    """KEY_UP action configuration."""

    keys: list[str]
    release_modifiers_first: bool | None = Field(None, alias="releaseModifiersFirst")

    model_config = {"populate_by_name": True}


class HotkeyActionConfig(BaseModel):
    """HOTKEY action configuration."""

    hotkey: str
    hold_duration: int | None = Field(None, alias="holdDuration")
    parse_string: bool | None = Field(None, alias="parseString")

    model_config = {"populate_by_name": True}
