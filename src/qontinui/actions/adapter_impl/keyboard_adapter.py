"""Keyboard action adapters following Single Responsibility Principle.

Separates keyboard-specific functionality from other adapter concerns.
"""

from abc import ABC, abstractmethod

from .adapter_result import AdapterResult


class KeyboardAdapter(ABC):
    """Abstract base for keyboard-specific actions."""

    @abstractmethod
    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key."""
        pass

    @abstractmethod
    def key_up(self, key: str) -> AdapterResult:
        """Release key."""
        pass

    @abstractmethod
    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up)."""
        pass

    @abstractmethod
    def type_character(self, char: str) -> AdapterResult:
        """Type single character."""
        pass


class HALKeyboardAdapter(KeyboardAdapter):
    """HAL implementation of keyboard actions."""

    def __init__(self, input_controller) -> None:
        """Initialize HAL keyboard adapter.

        Args:
            input_controller: HAL input controller
        """
        self.input_controller = input_controller

    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key."""
        try:
            success = self.input_controller.key_down(key)
            if success:
                return AdapterResult(success=True, data=key)
            return AdapterResult(success=False, error="Key down failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_up(self, key: str) -> AdapterResult:
        """Release key."""
        try:
            success = self.input_controller.key_up(key)
            if success:
                return AdapterResult(success=True, data=key)
            return AdapterResult(success=False, error="Key up failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up)."""
        try:
            success = self.input_controller.key_press(key)
            if success:
                return AdapterResult(success=True, data=key)
            return AdapterResult(success=False, error="Key press failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def type_character(self, char: str) -> AdapterResult:
        """Type single character."""
        try:
            if len(char) != 1:
                return AdapterResult(success=False, error="Must be single character")
            success = self.input_controller.type_text(char)
            if success:
                return AdapterResult(success=True, data=char)
            return AdapterResult(success=False, error="Type character failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))


class SeleniumKeyboardAdapter(KeyboardAdapter):
    """Selenium implementation of keyboard actions."""

    def __init__(self, action_chains) -> None:
        """Initialize Selenium keyboard adapter.

        Args:
            action_chains: Selenium ActionChains instance
        """
        self.actions = action_chains

    def key_down(self, key: str) -> AdapterResult:
        """Press key in browser context."""
        try:
            from selenium.webdriver.common.keys import Keys

            selenium_key = getattr(Keys, key.upper(), key)
            self.actions.key_down(selenium_key)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_up(self, key: str) -> AdapterResult:
        """Release key in browser context."""
        try:
            from selenium.webdriver.common.keys import Keys

            selenium_key = getattr(Keys, key.upper(), key)
            self.actions.key_up(selenium_key)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_press(self, key: str) -> AdapterResult:
        """Press key in browser context."""
        try:
            from selenium.webdriver.common.keys import Keys

            selenium_key = getattr(Keys, key.upper(), key)
            self.actions.send_keys(selenium_key)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def type_character(self, char: str) -> AdapterResult:
        """Type character in browser context."""
        try:
            if len(char) != 1:
                return AdapterResult(success=False, error="Must be single character")
            self.actions.send_keys(char)
            self.actions.perform()
            return AdapterResult(success=True, data=char)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
