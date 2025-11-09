# Qontinui

[![PyPI version](https://badge.fury.io/py/qontinui.svg)](https://badge.fury.io/py/qontinui)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library for model-based GUI automation with intelligent state management and visual recognition.

## Research Foundation

Based on [**Model-based GUI Automation**](https://link.springer.com/article/10.1007/s10270-025-01319-9) published in Springer's Software and Systems Modeling journal (October 2025).

The research provides:
- Mathematical proof of complexity reduction (exponential → polynomial)
- First testable approach to GUI automation (unit tests, integration tests)
- Formal framework for robust visual APIs for RL agents
- Enables reliable dataset generation for AI training

## Overview

Qontinui enables building robust GUI automation through:
- **Model-based state management** using [MultiState](https://github.com/qontinui/multistate) | [Docs](https://qontinui.github.io/multistate/)
- **Visual recognition** with OpenCV template matching
- **JSON configuration** for defining automation workflows
- **Cross-platform support** (Windows, macOS, Linux)

Qontinui is a Python port of [Brobot](https://github.com/jspinak/brobot), a Java library for GUI automation (2018-2024).

## Installation

### From PyPI (Recommended)

```bash
pip install qontinui
```

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/jspinak/qontinui.git
cd qontinui

# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

### Dependencies

Qontinui requires:
- **[MultiState](https://github.com/jspinak/multistate)** - Multi-state state management
- **OpenCV** - Image template matching
- **PyAutoGUI/pynput** - Input control

## Quick Start

### JSON Configuration

Create an automation configuration in JSON:

```json
{
  "version": "1.0",
  "states": [
    {
      "name": "LoginScreen",
      "stateImages": [
        {
          "imageId": "login_button",
          "threshold": 0.9
        }
      ]
    }
  ],
  "processes": [
    {
      "name": "Login",
      "actions": [
        {
          "type": "CLICK",
          "target": {"type": "image", "imageId": "login_button"}
        },
        {
          "type": "TYPE",
          "text": "username@example.com"
        }
      ]
    }
  ]
}
```

### Python API

```python
from qontinui.json_executor import JSONRunner

# Initialize runner
runner = JSONRunner()

# Load configuration
runner.load_configuration("automation_config.json")

# Execute automation
success = runner.run(monitor_index=0)
```

### Desktop Application

Use [Qontinui Runner](https://github.com/qontinui/qontinui-runner) for a GUI interface to create and run automations.

## Architecture

```
qontinui/
├── src/qontinui/
│   ├── json_executor/       # JSON configuration execution
│   ├── model/               # State, Transition, Image models
│   ├── hal/                 # Hardware Abstraction Layer
│   └── multistate_adapter/  # MultiState integration
```

## Features

### Current
- ✅ JSON-based automation configuration
- ✅ Template-based image matching
- ✅ Multi-state state management
- ✅ Process and state machine execution modes
- ✅ Cross-platform input control (PyAutoGUI, pynput)
- ✅ Hardware abstraction layer for multiple backends

### Planned
- 🔄 AI-enhanced visual recognition (SAM, CLIP)
- 🔄 Domain-specific language (DSL)
- 🔄 Advanced Brobot migration tools
- 🔄 Cloud execution via qontinui-web

## Testing

Run the test suite:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/qontinui

# Run specific test file
poetry run pytest tests/json_executor/test_json_runner.py
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

Qontinui is a faithful port of Brobot. When contributing, please preserve Brobot's architecture and behavior.

## Documentation

- **GitHub**: [github.com/qontinui/qontinui](https://github.com/qontinui/qontinui)
- **Issues**: [GitHub Issues](https://github.com/qontinui/qontinui/issues)
- **MultiState Docs**: [qontinui.github.io/multistate](https://qontinui.github.io/multistate/)
- **Research Paper**: [Springer SoSyM](https://link.springer.com/article/10.1007/s10270-025-01319-9)

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Brobot](https://github.com/jspinak/brobot) - The original Java library
- [MultiState](https://github.com/jspinak/multistate) - Multi-state state management
- [PyAutoGUI](https://github.com/asweigart/pyautogui) - GUI automation
- [OpenCV](https://opencv.org/) - Computer vision

## Related Projects

- **[qontinui-runner](https://github.com/qontinui/qontinui-runner)** - Desktop application (Rust/TypeScript)
- **[qontinui-api](https://github.com/qontinui/qontinui-api)** - REST API bridge (enables custom frontends)
- **[qontinui-web](https://qontinui.com)** - Web-based visual builder (launching Feb 2026)
- **[multistate](https://github.com/qontinui/multistate)** - State management library | [Docs](https://qontinui.github.io/multistate/)
- **[Brobot](https://github.com/jspinak/brobot)** - Original Java implementation (2018-2024)
