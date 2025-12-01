# Runtime Extraction

Runtime extraction for live applications across different runtime environments.

## Overview

The runtime extraction module provides extractors that connect to live applications and extract their GUI state through browser automation, DOM inspection, and interaction simulation. This complements static analysis by discovering the actual runtime behavior of applications.

## Architecture

```
runtime/
├── base.py                 # RuntimeExtractor abstract base class
├── types.py               # Core types and data structures
├── playwright/            # Playwright-based web extractor
│   ├── extractor.py      # PlaywrightExtractor implementation
│   └── __init__.py
├── tauri/                 # Tauri application extractor
│   ├── extractor.py      # TauriExtractor implementation
│   ├── mock.py           # Tauri API mocks (Python)
│   └── __init__.py
└── injection/             # JavaScript injection scripts
    ├── tauri_mock.js     # Full Tauri API mock script
    └── __init__.py
```

## Extractors

### PlaywrightExtractor

Extracts UI state from web applications using Playwright.

**Features:**
- Reuses existing `ElementClassifier` and `RegionDetector` from web extraction
- Extracts interactive elements (buttons, inputs, links, etc.)
- Detects UI regions (navigation, modals, sidebars, etc.)
- Captures screenshots
- Simulates user interactions

**Usage:**

```python
from qontinui.extraction.runtime import (
    PlaywrightExtractor,
    ExtractionTarget,
    RuntimeType,
)

# Create target
target = ExtractionTarget(
    runtime_type=RuntimeType.WEB,
    url="https://example.com",
    viewport=(1920, 1080),
    headless=True,
)

# Extract state
async with PlaywrightExtractor() as extractor:
    session = await extractor.start_session(target)

    # Extract current state
    capture = await extractor.extract_current_state()
    print(f"Found {len(capture.elements)} elements")
    print(f"Found {len(capture.states)} states")

    # Perform action and capture result
    new_capture = await extractor.perform_action(
        action_type="click",
        target_selector="button.submit",
    )

    # Navigate to different route
    await extractor.navigate_to_route("https://example.com/about")
    about_capture = await extractor.extract_current_state()
```

### TauriExtractor

Extracts UI state from Tauri applications by connecting to their dev server and injecting Tauri API mocks.

**How it works:**
1. Starts the Tauri dev server (if `app_dev_command` provided)
2. Connects to it via Playwright
3. Injects Tauri API mocks so the app runs in browser
4. Uses Playwright's DOM extraction capabilities

**Features:**
- Automatic Tauri API mocking
- Custom mock responses
- Dev server management
- Event simulation

**Usage:**

```python
from qontinui.extraction.runtime import (
    TauriExtractor,
    ExtractionTarget,
    RuntimeType,
)

# Create target
target = ExtractionTarget(
    runtime_type=RuntimeType.TAURI,
    url="http://localhost:1420",  # Tauri dev server
    app_dev_command="npm run tauri dev",  # Optional: auto-start
    tauri_config_path="./src-tauri/tauri.conf.json",
    tauri_mocks={
        "get_user": {"name": "Test User", "id": 123},
        "load_data": {"items": [1, 2, 3]},
    },
    viewport=(1280, 720),
    headless=False,
)

# Extract state
async with TauriExtractor() as extractor:
    session = await extractor.start_session(target)

    # Extract current state (with Tauri mocks active)
    capture = await extractor.extract_current_state()

    # Inject custom mock at runtime
    await extractor.inject_custom_mock(
        "get_settings",
        {"theme": "dark", "language": "en"}
    )

    # Simulate Tauri event
    await extractor.simulate_tauri_event(
        "notification",
        {"title": "Test", "body": "Hello!"}
    )
```

## Core Types

### ExtractionTarget

Configuration for connecting to a target application.

```python
@dataclass
class ExtractionTarget:
    runtime_type: RuntimeType  # WEB, TAURI, ELECTRON, NATIVE
    url: str | None = None
    app_path: str | None = None
    app_dev_command: str | None = None
    viewport: tuple[int, int] = (1920, 1080)
    headless: bool = True
    auth_cookies: dict[str, str] = field(default_factory=dict)
    tauri_config_path: str | None = None
    tauri_mocks: dict[str, Any] = field(default_factory=dict)
```

### RuntimeStateCapture

A snapshot of the application state at a point in time.

```python
@dataclass
class RuntimeStateCapture:
    capture_id: str
    timestamp: datetime
    elements: list[ExtractedElement]
    states: list[ExtractedState]
    screenshot_path: Path | None
    url: str | None
    title: str | None
    viewport: tuple[int, int]
    scroll_position: tuple[int, int]
```

### RuntimeExtractionSession

A complete extraction session with multiple captures.

```python
@dataclass
class RuntimeExtractionSession:
    session_id: str
    target: ExtractionTarget
    started_at: datetime
    completed_at: datetime | None
    captures: list[RuntimeStateCapture]
    transitions: list[ExtractedTransition]
    storage_dir: Path | None
```

## Tauri API Mocking

The Tauri extractor provides comprehensive mocking of Tauri APIs:

### Available Mock APIs

- **Core**: `invoke()`, event system
- **Window**: get/set window properties, show/hide, minimize/maximize
- **Dialog**: file dialogs, message boxes, confirmations
- **Filesystem**: read/write files, directory operations
- **HTTP**: network requests
- **Shell**: open URLs, execute commands
- **Notification**: send notifications
- **Clipboard**: read/write clipboard
- **Path**: access system directories

### Custom Mocks

You can provide custom mock responses:

**Python:**
```python
from qontinui.extraction.runtime.tauri import generate_mock_script

mocks = {
    "get_user_profile": {
        "name": "John Doe",
        "email": "john@example.com",
        "role": "admin"
    },
    "load_settings": {
        "theme": "dark",
        "notifications": True
    }
}

script = generate_mock_script(mocks)
```

**JavaScript injection:**
```javascript
// Inject mocks before Tauri app loads
window.__TAURI_MOCKS__ = {
    'get_data': async (args) => {
        return { items: [1, 2, 3], total: 3 };
    },
    'save_data': true
};
```

### Standalone Mock Script

The full Tauri mock is available as a standalone JavaScript file:

```python
from qontinui.extraction.runtime.injection import (
    TAURI_MOCK_JS_PATH,
    get_tauri_mock_script
)

# Get path to JavaScript file
print(TAURI_MOCK_JS_PATH)

# Get script contents
script = get_tauri_mock_script()
```

Or use directly with Playwright:

```python
await page.add_init_script(path=str(TAURI_MOCK_JS_PATH))
```

## Integration with Existing Code

### Reusing Web Extraction Components

The `PlaywrightExtractor` reuses the existing web extraction components:

- `ElementClassifier` from `qontinui.extraction.web.element_classifier`
- `RegionDetector` from `qontinui.extraction.web.region_detector`
- Models from `qontinui.extraction.web.models`

This ensures consistency between static web extraction and runtime extraction.

### Data Compatibility

All extractors produce the same data structures:

- `ExtractedElement` - GUI elements
- `ExtractedState` - UI states/regions
- `ExtractedTransition` - State transitions
- `BoundingBox` - Element positions

These can be directly used with the rest of the Qontinui pipeline.

## Future Extractors

### ElectronExtractor (Planned)

Similar to TauriExtractor, will mock Electron APIs:
- IPC communication
- Native modules
- File system access
- Window management

### NativeExtractor (Planned)

For native desktop applications:
- Windows: UI Automation API
- macOS: Accessibility API
- Linux: AT-SPI

## Examples

### Complete Web Extraction

```python
from qontinui.extraction.runtime import (
    PlaywrightExtractor,
    ExtractionTarget,
    RuntimeType,
)

async def extract_web_app():
    target = ExtractionTarget(
        runtime_type=RuntimeType.WEB,
        url="https://example.com/app",
        viewport=(1920, 1080),
        auth_cookies={"session": "abc123"},
    )

    async with PlaywrightExtractor() as extractor:
        session = await extractor.start_session(
            target,
            storage_dir=Path("./extraction_output")
        )

        # Extract initial state
        home = await extractor.extract_current_state()

        # Navigate and extract
        await extractor.navigate_to_route("https://example.com/app/settings")
        settings = await extractor.extract_current_state()

        # Interact and extract
        profile = await extractor.perform_action(
            "click",
            "a[href='/profile']"
        )

        # Get completed session
        completed = await extractor.end_session()
        print(f"Captured {len(completed.captures)} states")

        return completed

# Run extraction
import asyncio
session = asyncio.run(extract_web_app())
```

### Tauri App with Custom Mocks

```python
from qontinui.extraction.runtime import (
    TauriExtractor,
    ExtractionTarget,
    RuntimeType,
)

async def extract_tauri_app():
    target = ExtractionTarget(
        runtime_type=RuntimeType.TAURI,
        url="http://localhost:1420",
        app_dev_command="npm run tauri dev",
        tauri_config_path="./src-tauri/tauri.conf.json",
        tauri_mocks={
            # Mock backend API calls
            "fetch_todos": [
                {"id": 1, "title": "Task 1", "done": False},
                {"id": 2, "title": "Task 2", "done": True},
            ],
            "save_todo": True,
            "delete_todo": True,
        },
        headless=False,
    )

    async with TauriExtractor() as extractor:
        session = await extractor.start_session(target)

        # Wait for app to initialize
        await extractor.wait_for_stability(2000)

        # Extract state
        capture = await extractor.extract_current_state()

        # Simulate adding a todo
        await extractor.inject_custom_mock(
            "add_todo",
            {"id": 3, "title": "New Task", "done": False}
        )

        # Click add button
        await extractor.perform_action("click", "button.add-todo")
        await extractor.extract_current_state()

        return await extractor.end_session()

# Run extraction
import asyncio
session = asyncio.run(extract_tauri_app())
```

## Testing

Run tests for runtime extraction:

```bash
# All runtime extraction tests
pytest tests/extraction/runtime/

# Specific extractor tests
pytest tests/extraction/runtime/test_playwright_extractor.py
pytest tests/extraction/runtime/test_tauri_extractor.py

# With coverage
pytest --cov=src/qontinui/extraction/runtime tests/extraction/runtime/
```

## Contributing

When adding new extractors:

1. Inherit from `RuntimeExtractor`
2. Implement all abstract methods
3. Add to `runtime/__init__.py`
4. Add tests in `tests/extraction/runtime/`
5. Update this README

## License

Part of the Qontinui project. See main LICENSE file.
