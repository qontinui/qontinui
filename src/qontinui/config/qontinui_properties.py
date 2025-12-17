"""Qontinui properties - configuration framework using Pydantic.

Centralized configuration using Pydantic for type safety and validation.
Properties are organized into themed groups for better maintainability.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Import individual config classes from property groups
from .property_groups.core_properties import AutomationConfig, CoreConfig, StartupConfig
from .property_groups.debug_properties import (
    ConsoleActionConfig,
    GuiAccessConfig,
    TestingConfig,
)
from .property_groups.display_properties import (
    CaptureConfig,
    DpiConfig,
    HighlightConfig,
    IllustrationConfig,
    MonitorConfig,
)
from .property_groups.input_properties import MouseConfig, SikuliConfig
from .property_groups.logging_properties import LoggingConfig
from .property_groups.output_properties import (
    DatasetConfig,
    RecordingConfig,
    ScreenshotConfig,
)
from .property_groups.timing_properties import MockConfig
from .property_groups.vision_properties import (
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

    Example usage:
        # Load from environment variables
        config = QontinuiProperties()

        # Load from dict
        config = QontinuiProperties(**config_dict)

        # Load from YAML
        import yaml
        with open('config.yaml') as f:
            config = QontinuiProperties(**yaml.safe_load(f))

        # Access nested properties
        print(config.mouse.move_delay)

        # Update properties (with validation)
        config.core.mock = True
    """

    model_config = ConfigDict(validate_assignment=True)

    # Delegated property access through composition
    core: CoreConfig = Field(
        default_factory=CoreConfig, description="Core framework settings"
    )
    startup: StartupConfig = Field(
        default_factory=StartupConfig, description="Startup configuration"
    )
    automation: AutomationConfig = Field(
        default_factory=AutomationConfig, description="Automation failure handling"
    )

    mouse: MouseConfig = Field(
        default_factory=MouseConfig, description="Mouse action configuration"
    )
    sikuli: SikuliConfig = Field(
        default_factory=SikuliConfig, description="SikuliX integration settings"
    )

    autoscaling: AutoScalingConfig = Field(
        default_factory=AutoScalingConfig, description="Automatic pattern scaling"
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig, description="Color analysis settings"
    )
    image_debug: ImageDebugConfig = Field(
        default_factory=ImageDebugConfig, description="Image debugging configuration"
    )

    mock: MockConfig = Field(
        default_factory=MockConfig, description="Mock mode timing configuration"
    )

    screenshot: ScreenshotConfig = Field(
        default_factory=ScreenshotConfig, description="Screenshot and history settings"
    )
    recording: RecordingConfig = Field(
        default_factory=RecordingConfig, description="Screen recording settings"
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig, description="AI dataset generation settings"
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Comprehensive logging configuration"
    )

    testing: TestingConfig = Field(
        default_factory=TestingConfig, description="Test execution settings"
    )
    gui_access: GuiAccessConfig = Field(
        default_factory=GuiAccessConfig, description="GUI access verification"
    )
    console: ConsoleActionConfig = Field(
        default_factory=ConsoleActionConfig, description="Console action reporting"
    )

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
        """Export configuration to YAML format.

        Args:
            path: Optional path to save YAML file

        Returns:
            YAML string representation
        """
        import yaml

        yaml_str = yaml.dump(self.model_dump(), default_flow_style=False)

        if path:
            path.write_text(yaml_str)

        return yaml_str

    def to_env_file(self, path: Path | None = None) -> str:
        """Export configuration to .env format.

        Args:
            path: Optional path to save .env file

        Returns:
            Environment variable format string
        """
        lines = []

        def flatten_dict(d, prefix="QONTINUI"):
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
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            QontinuiProperties instance
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env_file(cls, path: Path) -> "QontinuiProperties":
        """Load configuration from .env file.

        Args:
            path: Path to .env file

        Returns:
            QontinuiProperties instance
        """
        from dotenv import dotenv_values

        env_vars = dotenv_values(path)

        # Parse environment variables into nested dict
        config: dict[str, Any] = {}
        for key, value in env_vars.items():
            if key.startswith("QONTINUI__") and value is not None:
                parts = key[10:].lower().split("__")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Convert value types
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
