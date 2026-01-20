"""Tests for HealingConfig."""

import sys
from pathlib import Path

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
