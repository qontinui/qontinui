"""Tests for UI-TARS configuration settings."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.extraction.runtime.uitars.config import UITARSSettings  # noqa: E402


class TestUITARSSettings:
    """Test UITARSSettings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = UITARSSettings()

        # Provider defaults
        assert settings.provider == "local_transformers"
        assert settings.model_size == "2B"

        # Quantization defaults
        assert settings.quantization == "int4"
        assert settings.device == "auto"
        assert settings.torch_dtype == "auto"

        # Execution defaults (local is default)
        assert settings.execution_mode == "local"
        assert settings.uitars_fallback_enabled is True
        assert settings.confidence_threshold == 0.7

        # Exploration defaults
        assert settings.max_exploration_steps == 50
        assert settings.exploration_timeout_seconds == 600
        assert settings.save_screenshots is True

        # Inference defaults
        assert settings.max_new_tokens == 512
        assert settings.temperature == 0.0

    def test_cloud_provider_settings(self):
        """Test settings for cloud provider."""
        settings = UITARSSettings(
            provider="cloud",
            huggingface_endpoint="https://test.endpoint.huggingface.cloud",
            huggingface_api_token="hf_test_token",
            model_size="7B",
        )

        assert settings.provider == "cloud"
        assert settings.huggingface_endpoint == "https://test.endpoint.huggingface.cloud"
        assert settings.huggingface_api_token == "hf_test_token"
        assert settings.model_size == "7B"
        assert settings.is_cloud() is True
        assert settings.is_local() is False

    def test_local_vllm_settings(self):
        """Test settings for vLLM provider."""
        settings = UITARSSettings(
            provider="local_vllm",
            vllm_server_url="http://localhost:8080",
            vllm_model_name="uitars-2b",
        )

        assert settings.provider == "local_vllm"
        assert settings.vllm_server_url == "http://localhost:8080"
        assert settings.vllm_model_name == "uitars-2b"
        assert settings.is_local() is True
        assert settings.is_cloud() is False

    def test_local_transformers_with_quantization(self):
        """Test local transformers with int4 quantization."""
        settings = UITARSSettings(
            provider="local_transformers",
            model_size="2B",
            quantization="int4",
            device="cuda",
        )

        assert settings.provider == "local_transformers"
        assert settings.quantization == "int4"
        assert settings.needs_quantization() is True
        assert settings.device == "cuda"

    def test_no_quantization(self):
        """Test settings without quantization."""
        settings = UITARSSettings(
            quantization="none",
        )

        assert settings.quantization == "none"
        assert settings.needs_quantization() is False

    def test_int8_quantization(self):
        """Test int8 quantization setting."""
        settings = UITARSSettings(quantization="int8")

        assert settings.quantization == "int8"
        assert settings.needs_quantization() is True

    def test_get_model_id_2b(self):
        """Test model ID for 2B model."""
        settings = UITARSSettings(model_size="2B")
        assert settings.get_model_id() == "ByteDance-Seed/UI-TARS-2B-SFT"

    def test_get_model_id_7b(self):
        """Test model ID for 7B model."""
        settings = UITARSSettings(model_size="7B")
        assert settings.get_model_id() == "ByteDance-Seed/UI-TARS-7B-SFT"

    def test_get_model_id_72b(self):
        """Test model ID for 72B model."""
        settings = UITARSSettings(model_size="72B")
        assert settings.get_model_id() == "ByteDance-Seed/UI-TARS-72B-SFT"

    def test_get_model_id_custom(self):
        """Test custom model ID."""
        settings = UITARSSettings(model_size="custom/model-name")
        assert settings.get_model_id() == "custom/model-name"

    def test_execution_mode_local(self):
        """Test local execution mode (default)."""
        settings = UITARSSettings(execution_mode="local")
        assert settings.execution_mode == "local"

    def test_execution_mode_uitars(self):
        """Test UI-TARS execution mode."""
        settings = UITARSSettings(execution_mode="uitars")
        assert settings.execution_mode == "uitars"

    def test_execution_mode_hybrid(self):
        """Test hybrid execution mode."""
        settings = UITARSSettings(execution_mode="hybrid")
        assert settings.execution_mode == "hybrid"

    def test_confidence_threshold(self):
        """Test confidence threshold setting."""
        settings = UITARSSettings(confidence_threshold=0.85)
        assert settings.confidence_threshold == 0.85

    def test_exploration_settings(self):
        """Test exploration-specific settings."""
        settings = UITARSSettings(
            max_exploration_steps=100,
            exploration_timeout_seconds=1200,
            step_timeout_seconds=60.0,
            save_screenshots=False,
            screenshot_format="jpg",
        )

        assert settings.max_exploration_steps == 100
        assert settings.exploration_timeout_seconds == 1200
        assert settings.step_timeout_seconds == 60.0
        assert settings.save_screenshots is False
        assert settings.screenshot_format == "jpg"

    def test_inference_settings(self):
        """Test inference parameter settings."""
        settings = UITARSSettings(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        assert settings.max_new_tokens == 256
        assert settings.temperature == 0.7
        assert settings.top_p == 0.9

    def test_custom_system_prompt(self):
        """Test custom system prompt."""
        custom_prompt = "You are a custom automation assistant."
        settings = UITARSSettings(system_prompt=custom_prompt)
        assert settings.system_prompt == custom_prompt

    def test_is_local_for_transformers(self):
        """Test is_local for local_transformers."""
        settings = UITARSSettings(provider="local_transformers")
        assert settings.is_local() is True
        assert settings.is_cloud() is False

    def test_is_local_for_vllm(self):
        """Test is_local for local_vllm."""
        settings = UITARSSettings(provider="local_vllm")
        assert settings.is_local() is True
        assert settings.is_cloud() is False


class TestUITARSSettingsEnvironment:
    """Test settings loading from environment variables."""

    def test_load_provider_from_env(self):
        """Test loading provider from environment."""
        with patch.dict(os.environ, {"QONTINUI_UITARS_PROVIDER": "cloud"}):
            settings = UITARSSettings()
            assert settings.provider == "cloud"

    def test_load_model_size_from_env(self):
        """Test loading model_size from environment."""
        with patch.dict(os.environ, {"QONTINUI_UITARS_MODEL_SIZE": "7B"}):
            settings = UITARSSettings()
            assert settings.model_size == "7B"

    def test_load_quantization_from_env(self):
        """Test loading quantization from environment."""
        with patch.dict(os.environ, {"QONTINUI_UITARS_QUANTIZATION": "int8"}):
            settings = UITARSSettings()
            assert settings.quantization == "int8"

    def test_load_execution_mode_from_env(self):
        """Test loading execution_mode from environment."""
        with patch.dict(os.environ, {"QONTINUI_UITARS_EXECUTION_MODE": "hybrid"}):
            settings = UITARSSettings()
            assert settings.execution_mode == "hybrid"

    def test_load_confidence_threshold_from_env(self):
        """Test loading confidence_threshold from environment."""
        with patch.dict(os.environ, {"QONTINUI_UITARS_CONFIDENCE_THRESHOLD": "0.9"}):
            settings = UITARSSettings()
            assert settings.confidence_threshold == 0.9

    def test_load_huggingface_settings_from_env(self):
        """Test loading HuggingFace settings from environment."""
        env_vars = {
            "QONTINUI_UITARS_HUGGINGFACE_ENDPOINT": "https://test.endpoint.com",
            "QONTINUI_UITARS_HUGGINGFACE_API_TOKEN": "hf_test_token",
        }
        with patch.dict(os.environ, env_vars):
            settings = UITARSSettings()
            assert settings.huggingface_endpoint == "https://test.endpoint.com"
            assert settings.huggingface_api_token == "hf_test_token"

    def test_load_vllm_settings_from_env(self):
        """Test loading vLLM settings from environment."""
        env_vars = {
            "QONTINUI_UITARS_VLLM_SERVER_URL": "http://localhost:9000",
            "QONTINUI_UITARS_VLLM_MODEL_NAME": "custom-model",
        }
        with patch.dict(os.environ, env_vars):
            settings = UITARSSettings()
            assert settings.vllm_server_url == "http://localhost:9000"
            assert settings.vllm_model_name == "custom-model"

    def test_load_exploration_settings_from_env(self):
        """Test loading exploration settings from environment."""
        env_vars = {
            "QONTINUI_UITARS_MAX_EXPLORATION_STEPS": "75",
            "QONTINUI_UITARS_EXPLORATION_TIMEOUT_SECONDS": "900",
            "QONTINUI_UITARS_SAVE_SCREENSHOTS": "false",
        }
        with patch.dict(os.environ, env_vars):
            settings = UITARSSettings()
            assert settings.max_exploration_steps == 75
            assert settings.exploration_timeout_seconds == 900
            assert settings.save_screenshots is False

    def test_explicit_value_overrides_env(self):
        """Test that explicit values override environment variables."""
        with patch.dict(os.environ, {"QONTINUI_UITARS_PROVIDER": "cloud"}):
            settings = UITARSSettings(provider="local_transformers")
            assert settings.provider == "local_transformers"
