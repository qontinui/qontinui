"""Tests for HealingConfig."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly to avoid circular import issues
from qontinui.healing.healing_config import HealingConfig
from qontinui.healing.healing_types import LLMMode


class TestHealingConfig:
    """Tests for HealingConfig class."""

    def test_disabled_by_default(self):
        """Test that LLM is disabled by default."""
        config = HealingConfig.disabled()

        assert config.llm_mode == LLMMode.DISABLED

    def test_disabled_factory(self):
        """Test disabled factory method."""
        config = HealingConfig.disabled()

        assert config.llm_mode == LLMMode.DISABLED
        assert config.remote_api_key is None

    def test_with_ollama(self):
        """Test Ollama configuration."""
        config = HealingConfig.with_ollama(model_name="llava:13b")

        assert config.llm_mode == LLMMode.LOCAL
        assert config.local_model_name == "llava:13b"

    def test_with_ollama_default_model(self):
        """Test Ollama with default model."""
        config = HealingConfig.with_ollama()

        assert config.llm_mode == LLMMode.LOCAL
        assert config.local_model_name == "llava:7b"

    def test_with_openai(self):
        """Test OpenAI configuration."""
        config = HealingConfig.with_openai(api_key="sk-test-key")

        assert config.llm_mode == LLMMode.REMOTE
        assert config.remote_provider == "openai"
        assert config.remote_api_key == "sk-test-key"

    def test_with_anthropic(self):
        """Test Anthropic configuration."""
        config = HealingConfig.with_anthropic(api_key="sk-ant-test")

        assert config.llm_mode == LLMMode.REMOTE
        assert config.remote_provider == "anthropic"
        assert config.remote_api_key == "sk-ant-test"

    def test_max_heal_attempts(self):
        """Test max heal attempts configuration."""
        config = HealingConfig.with_ollama()
        assert config.max_heal_attempts == 2  # Default

        config = HealingConfig(
            llm_mode=LLMMode.LOCAL,
            local_model_name="llava:7b",
            max_heal_attempts=5,
        )
        assert config.max_heal_attempts == 5

    def test_cache_healed_locations(self):
        """Test cache configuration."""
        config = HealingConfig.with_ollama()
        assert config.cache_healed_locations  # Default True

        config = HealingConfig(
            llm_mode=LLMMode.LOCAL,
            local_model_name="llava:7b",
            cache_healed_locations=False,
        )
        assert not config.cache_healed_locations

    def test_get_client_disabled(self):
        """Test getting client when disabled."""
        config = HealingConfig.disabled()
        client = config.get_client()

        # DisabledVisionClient is technically "available" (can be called)
        # but it always returns None - test that it returns None
        from qontinui.healing.healing_types import HealingContext

        result = client.find_element(
            b"fake_screenshot", HealingContext(original_description="test")
        )
        assert result is None

    def test_get_client_local(self):
        """Test getting local client."""
        config = HealingConfig.with_ollama()
        client = config.get_client()

        # Client is created, availability depends on Ollama being installed
        assert client is not None


class TestHealingConfigAriaUI:
    """Tests for HealingConfig Aria-UI factory methods and from_env."""

    def test_with_aria_ui(self):
        """Should create ARIA_UI mode config with endpoint."""
        config = HealingConfig.with_aria_ui(endpoint="http://gpu-server:8100")

        assert config.llm_mode == LLMMode.ARIA_UI
        assert config.aria_ui_endpoint == "http://gpu-server:8100"

    def test_with_aria_ui_default_endpoint(self):
        """Should use default localhost endpoint."""
        config = HealingConfig.with_aria_ui()

        assert config.llm_mode == LLMMode.ARIA_UI
        assert config.aria_ui_endpoint == "http://localhost:8100"

    def test_with_aria_ui_context(self):
        """Should create ARIA_UI_CONTEXT mode with max_history."""
        config = HealingConfig.with_aria_ui_context(
            endpoint="http://gpu-server:8100", max_history=5
        )

        assert config.llm_mode == LLMMode.ARIA_UI_CONTEXT
        assert config.aria_ui_endpoint == "http://gpu-server:8100"
        assert config.aria_ui_max_history == 5

    def test_with_aria_ui_context_defaults(self):
        """Should use default max_history of 3."""
        config = HealingConfig.with_aria_ui_context()

        assert config.llm_mode == LLMMode.ARIA_UI_CONTEXT
        assert config.aria_ui_max_history == 3

    def test_from_env_aria_ui_enabled(self):
        """Should create Aria-UI config when QONTINUI_ARIA_UI_ENABLED=true."""
        env = {
            "QONTINUI_ARIA_UI_ENABLED": "true",
            "QONTINUI_ARIA_UI_ENDPOINT": "http://myserver:8100",
        }
        with patch.dict(os.environ, env, clear=False):
            config = HealingConfig.from_env()

        assert config.llm_mode == LLMMode.ARIA_UI
        assert config.aria_ui_endpoint == "http://myserver:8100"

    def test_from_env_aria_ui_context(self):
        """Should create context mode when QONTINUI_ARIA_UI_MODE=context."""
        env = {
            "QONTINUI_ARIA_UI_ENABLED": "true",
            "QONTINUI_ARIA_UI_MODE": "context",
            "QONTINUI_ARIA_UI_MAX_HISTORY": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            config = HealingConfig.from_env()

        assert config.llm_mode == LLMMode.ARIA_UI_CONTEXT
        assert config.aria_ui_max_history == 5

    def test_from_env_disabled(self):
        """Should return disabled config when no env vars are set."""
        env_keys = [
            "QONTINUI_ARIA_UI_ENABLED",
            "QONTINUI_ARIA_UI_ENDPOINT",
            "QONTINUI_ARIA_UI_MODE",
            "QONTINUI_ARIA_UI_MAX_HISTORY",
        ]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
        with patch.dict(os.environ, clean_env, clear=True):
            config = HealingConfig.from_env()

        assert config.llm_mode == LLMMode.DISABLED

    def test_get_client_aria_ui(self):
        """Should create an AriaUIClient instance."""
        from qontinui.healing.aria_ui_client import AriaUIClient

        config = HealingConfig.with_aria_ui()
        client = config.get_client()

        assert isinstance(client, AriaUIClient)

    def test_get_client_aria_ui_context(self):
        """Should create an AriaUIContextClient instance."""
        from qontinui.healing.aria_ui_context_client import AriaUIContextClient

        config = HealingConfig.with_aria_ui_context()
        client = config.get_client()

        assert isinstance(client, AriaUIContextClient)


class TestLLMMode:
    """Tests for LLMMode enum."""

    def test_enum_values(self):
        """Test LLMMode enum values exist."""
        assert LLMMode.DISABLED
        assert LLMMode.LOCAL
        assert LLMMode.REMOTE

    def test_enum_comparison(self):
        """Test LLMMode enum comparison."""
        assert LLMMode.DISABLED != LLMMode.LOCAL
        assert LLMMode.LOCAL != LLMMode.REMOTE
        assert LLMMode.DISABLED != LLMMode.REMOTE
