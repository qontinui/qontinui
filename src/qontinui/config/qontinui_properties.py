"""Backwards-compatibility shim for QontinuiProperties.

The individual property-group ``*Config`` pydantic models now live canonically
in ``qontinui_schemas.config.property_groups.*`` so they can be shared between
the runner (qontinui) and the web tier (qontinui-web) without dragging heavy
qontinui deps into the web image.

``QontinuiProperties`` itself — the aggregator that composes every
``*Config`` plus YAML / .env serialization helpers — was never moved into
qontinui-schemas (the 0.3.0 release referenced in PR #12 did not ship it).
We therefore keep it defined here and re-export the leaf configs from
qontinui-schemas, so ``from qontinui.config.qontinui_properties import X``
continues to work for both ``QontinuiProperties`` and every ``*Config``.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Leaf configs (canonical source: qontinui_schemas.config.property_groups.*)
from qontinui_schemas.config.property_groups.core_properties import (  # noqa: F401
    AutomationConfig,
    CoreConfig,
    StartupConfig,
)
from qontinui_schemas.config.property_groups.debug_properties import (  # noqa: F401
    ConsoleActionConfig,
    GuiAccessConfig,
    TestingConfig,
)
from qontinui_schemas.config.property_groups.display_properties import (  # noqa: F401
    CaptureConfig,
    DpiConfig,
    HighlightConfig,
    IllustrationConfig,
    MonitorConfig,
)
from qontinui_schemas.config.property_groups.input_properties import (  # noqa: F401
    MouseConfig,
    SikuliConfig,
)
from qontinui_schemas.config.property_groups.logging_properties import (  # noqa: F401
    LoggingConfig,
)
from qontinui_schemas.config.property_groups.output_properties import (  # noqa: F401
    DatasetConfig,
    RecordingConfig,
    ScreenshotConfig,
)
from qontinui_schemas.config.property_groups.timing_properties import (  # noqa: F401
    MockConfig,
)
from qontinui_schemas.config.property_groups.vision_properties import (  # noqa: F401
    AnalysisConfig,
    AutoScalingConfig,
    ImageDebugConfig,
)


class QontinuiProperties(BaseModel):
    """Centralized configuration properties for the Qontinui framework.

    Facade that composes themed property groups for better organization
    and maintainability. Each group contains related configuration settings.

    Property Groups:
    - Core: Essential framework settings (core, startup, automation)
    - Input: Mouse/keyboard settings (mouse, sikuli)
    - Vision: Image finding (autoscaling, analysis, image_debug)
    - Timing: Mock execution timings (mock)
    - Output: Screenshots/recordings/datasets (screenshot, recording, dataset)
    - Logging: Logging configuration (logging)
    - Debug: Testing/debugging (testing, gui_access, console)
    - Display: Visual/monitor/capture (illustration, highlight, monitor, dpi, capture)
    """

    model_config = ConfigDict(validate_assignment=True)

    # Core
    core: CoreConfig = Field(
        default_factory=CoreConfig, description="Core framework settings"
    )
    startup: StartupConfig = Field(
        default_factory=StartupConfig, description="Startup configuration"
    )
    automation: AutomationConfig = Field(
        default_factory=AutomationConfig, description="Automation failure handling"
    )

    # Input
    mouse: MouseConfig = Field(
        default_factory=MouseConfig, description="Mouse action configuration"
    )
    sikuli: SikuliConfig = Field(
        default_factory=SikuliConfig, description="SikuliX integration settings"
    )

    # Vision
    autoscaling: AutoScalingConfig = Field(
        default_factory=AutoScalingConfig, description="Automatic pattern scaling"
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig, description="Color analysis settings"
    )
    image_debug: ImageDebugConfig = Field(
        default_factory=ImageDebugConfig, description="Image debugging configuration"
    )

    # Timing
    mock: MockConfig = Field(
        default_factory=MockConfig, description="Mock mode timing configuration"
    )

    # Output
    screenshot: ScreenshotConfig = Field(
        default_factory=ScreenshotConfig, description="Screenshot and history settings"
    )
    recording: RecordingConfig = Field(
        default_factory=RecordingConfig, description="Screen recording settings"
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig, description="AI dataset generation settings"
    )

    # Logging
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Comprehensive logging configuration"
    )

    # Debug
    testing: TestingConfig = Field(
        default_factory=TestingConfig, description="Test execution settings"
    )
    gui_access: GuiAccessConfig = Field(
        default_factory=GuiAccessConfig, description="GUI access verification"
    )
    console: ConsoleActionConfig = Field(
        default_factory=ConsoleActionConfig, description="Console action reporting"
    )

    # Display
    illustration: IllustrationConfig = Field(
        default_factory=IllustrationConfig, description="Action illustration settings"
    )
    highlight: HighlightConfig = Field(
        default_factory=HighlightConfig, description="Visual highlighting configuration"
    )
    monitor: MonitorConfig = Field(
        default_factory=MonitorConfig, description="Monitor configuration settings"
    )
    dpi: DpiConfig = Field(
        default_factory=DpiConfig, description="DPI and scaling configuration"
    )
    capture: CaptureConfig = Field(
        default_factory=CaptureConfig,
        description="Screen capture provider configuration",
    )

    def to_yaml(self, path: Path | None = None) -> str:
        """Export configuration to YAML format."""
        import yaml

        yaml_str = yaml.dump(self.model_dump(), default_flow_style=False)
        if path:
            path.write_text(yaml_str)
        return str(yaml_str)

    def to_env_file(self, path: Path | None = None) -> str:
        """Export configuration to .env format."""
        lines: list[str] = []

        def flatten_dict(d: dict[str, Any], prefix: str = "QONTINUI") -> None:
            for key, value in d.items():
                env_key = f"{prefix}__{key.upper()}"
                if isinstance(value, dict):
                    flatten_dict(value, env_key)
                elif isinstance(value, list):
                    lines.append(f"{env_key}={','.join(map(str, value))}")
                else:
                    lines.append(f"{env_key}={value}")

        flatten_dict(self.model_dump())
        env_str = "\n".join(lines)
        if path:
            path.write_text(env_str)
        return env_str

    @classmethod
    def from_yaml(cls, path: Path) -> "QontinuiProperties":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env_file(cls, path: Path) -> "QontinuiProperties":
        """Load configuration from .env file."""
        from dotenv import dotenv_values

        env_vars = dotenv_values(path)

        config: dict[str, Any] = {}
        for key, value in env_vars.items():
            if key.startswith("QONTINUI__") and value is not None:
                parts = key[10:].lower().split("__")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                parsed_value: Any = value
                if value is not None:
                    if value.lower() in ("true", "false"):
                        parsed_value = value.lower() == "true"
                    elif value.isdigit():
                        parsed_value = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        parsed_value = float(value)
                    elif "," in value:
                        parsed_value = value.split(",")

                current[parts[-1]] = parsed_value

        return cls(**config)


__all__ = [
    "QontinuiProperties",
    # Core
    "CoreConfig",
    "StartupConfig",
    "AutomationConfig",
    # Input
    "MouseConfig",
    "SikuliConfig",
    # Vision
    "AutoScalingConfig",
    "AnalysisConfig",
    "ImageDebugConfig",
    # Timing
    "MockConfig",
    # Output
    "ScreenshotConfig",
    "RecordingConfig",
    "DatasetConfig",
    # Logging
    "LoggingConfig",
    # Debug
    "TestingConfig",
    "GuiAccessConfig",
    "ConsoleActionConfig",
    # Display
    "IllustrationConfig",
    "HighlightConfig",
    "MonitorConfig",
    "DpiConfig",
    "CaptureConfig",
]
