# Qontinui Core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Qontinui Core is a Python library for model-based GUI automation with AI-enhanced perception. It provides semantic state recognition, auto-state discovery, and intelligent UI element detection using modern computer vision and machine learning techniques.

## Features

- **AI-Enhanced Perception**: Uses Segment Anything Model (SAM) and CLIP for intelligent UI element detection
- **Hybrid State Management**: Combines deterministic and probabilistic state management using pytransitions
- **Smart Element Matching**: Vector-based semantic matching with FAISS for efficient similarity search
- **DSL Support**: Domain-specific language for defining automation scripts
- **Brobot Migration**: Tools to migrate existing Brobot applications to Qontinui
- **Modular Architecture**: Clean separation of concerns with dedicated packages for actions, perception, and state management

## Installation

### Basic Installation

```bash
pip install qontinui
```

### Installation with SAM support

```bash
pip install qontinui[sam]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/your-org/qontinui-core.git
cd qontinui-core

# Install Poetry
pip install poetry

# Install dependencies
poetry install

# Install with all extras
poetry install --all-extras
```

## Quick Start

```python
from qontinui import QontinuiStateManager, State, Element
from qontinui.perception import ScreenSegmenter, ObjectVectorizer
from qontinui.actions import BasicActions

# Initialize components
state_manager = QontinuiStateManager()
segmenter = ScreenSegmenter(use_sam=True)
vectorizer = ObjectVectorizer()
actions = BasicActions()

# Define a state
login_state = State(
    name="login",
    elements=[
        Element(
            id="username_field",
            bbox=(100, 200, 200, 50),
            description="Username input field"
        ),
        Element(
            id="password_field",
            bbox=(100, 260, 200, 50),
            description="Password input field"
        ),
        Element(
            id="login_button",
            bbox=(150, 330, 100, 40),
            description="Login button"
        )
    ],
    min_elements=3
)

# Add state to manager
state_manager.add_state(login_state)

# Take screenshot and detect current state
screenshot = actions.screenshot()
segments = segmenter.segment_screen(screenshot)

# Check if login state is active
current_elements = [Element(...) for segment in segments]
state_manager.update_evidence(current_elements)

if "login" in state_manager.get_current_states():
    print("Login state detected!")
    # Perform login actions
    actions.click(150, 225)  # Click username field
    actions.type_text("user@example.com")
    actions.click(150, 285)  # Click password field
    actions.type_text("password")
    actions.click(200, 350)  # Click login button
```

## DSL Example

Qontinui supports a domain-specific language for defining automation scripts:

```qontinui
// Define states
state LoginPage {
    elements: [
        {id: "username_field", type: input, text: "Username"},
        {id: "password_field", type: input, text: "Password"},
        {id: "login_button", type: button, text: "Login"}
    ]
    min_elements: 3
}

state HomePage {
    elements: [
        {id: "welcome_text", type: text},
        {id: "logout_link", type: link, text: "Logout"}
    ]
    min_elements: 2
}

// Define transition
transition login_to_home {
    from: LoginPage
    to: HomePage
    action: click
    trigger: login_button
}

// Automation script
click(element=username_field);
type(text="user@example.com");
click(element=password_field);
type(text="password");
click(element=login_button);

assert current_state == HomePage;
```

## Migrating from Brobot

Convert your existing Brobot applications to Qontinui:

```bash
# Using the CLI tool
brobot-converter /path/to/brobot/app /path/to/output --use-sam --use-clip

# Using Python
from qontinui.migrations import BrobotConverter

converter = BrobotConverter(
    input_dir="/path/to/brobot/app",
    output_dir="/path/to/output",
    use_sam=True,
    use_clip=True
)

report = converter.convert_all()
print(f"Converted {report.converted_states} states")
```

## Architecture

```
qontinui-core/
├── src/qontinui/
│   ├── actions/          # GUI automation actions
│   │   ├── basic.py      # Basic PyAutoGUI actions
│   │   ├── advanced.py   # Advanced actions with retry logic
│   │   └── adapters.py   # Backend adapters (PyAutoGUI, Selenium)
│   ├── perception/       # Screen analysis and element detection
│   │   ├── segmentation.py    # SAM and OpenCV segmentation
│   │   ├── vectorization.py   # CLIP embeddings
│   │   └── matching.py        # FAISS similarity search
│   ├── state_management/ # Application state handling
│   │   ├── models.py     # State, Element, Transition models
│   │   ├── manager.py    # Hybrid state manager
│   │   └── traversal.py  # Graph traversal strategies
│   ├── dsl/             # Domain-specific language
│   │   └── parser.py    # Lark-based DSL parser
│   └── migrations/      # Migration tools
│       └── brobot_converter.py
```

## Configuration

### Environment Variables

```bash
# Enable debug logging
export QONTINUI_LOG_LEVEL=DEBUG

# Set SAM model checkpoint
export SAM_CHECKPOINT_PATH=/path/to/sam_vit_h.pth

# Set CLIP model
export CLIP_MODEL_NAME=openai/clip-vit-large-patch14
```

### Python Configuration

```python
import logging
from qontinui import QontinuiStateManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure state manager
manager = QontinuiStateManager(use_hierarchical=True)
manager.activation_threshold = 0.8
manager.deactivation_threshold = 0.3
manager.evidence_decay = 0.9
```

## Testing

Run the test suite:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/qontinui

# Run specific test file
poetry run pytest tests/state_management/test_manager.py

# Run with verbose output
poetry run pytest -v
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [PyAutoGUI](https://github.com/asweigart/pyautogui) for GUI automation
- [pytransitions](https://github.com/pytransitions/transitions) for state machine implementation
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search

## Support

- Documentation: [https://qontinui.github.io](https://qontinui.github.io)
- Issues: [GitHub Issues](https://github.com/your-org/qontinui-core/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/qontinui-core/discussions)

## Roadmap

- [ ] Week 1-2: Core state management and perception
- [ ] Week 3: DSL parser implementation
- [ ] Week 4: Enhanced Brobot converter
- [ ] Week 5-6: Web Builder skeleton (FastAPI/Next.js)
- [ ] Week 7-8: Desktop Runner (Tauri)
- [ ] Future: MCP server integration, advanced AI features