# Runtime Extraction Examples

Complete examples demonstrating the runtime extraction capabilities.

## Table of Contents

1. [Basic Web Extraction](#basic-web-extraction)
2. [Web App with Authentication](#web-app-with-authentication)
3. [Tauri App Extraction](#tauri-app-extraction)
4. [State Comparison](#state-comparison)
5. [Interactive Exploration](#interactive-exploration)

---

## Basic Web Extraction

Extract elements and states from a web application.

```python
import asyncio
from pathlib import Path
from qontinui.extraction.runtime import (
    PlaywrightExtractor,
    ExtractionTarget,
    RuntimeType,
)

async def basic_web_extraction():
    """Extract UI state from a web application."""

    # Configure target
    target = ExtractionTarget(
        runtime_type=RuntimeType.WEB,
        url="https://example.com",
        viewport=(1920, 1080),
        headless=True,
    )

    # Create extractor and start session
    extractor = PlaywrightExtractor()

    try:
        # Start session
        session = await extractor.start_session(
            target,
            storage_dir=Path("./output/web_extraction")
        )

        print(f"Session started: {session.session_id}")

        # Extract current state
        capture = await extractor.extract_current_state()

        print(f"Captured state: {capture.capture_id}")
        print(f"  Elements: {len(capture.elements)}")
        print(f"  States: {len(capture.states)}")
        print(f"  URL: {capture.url}")
        print(f"  Screenshot: {capture.screenshot_path}")

        # Print element types
        element_types = {}
        for elem in capture.elements:
            elem_type = elem.element_type.value
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

        print("\nElement types found:")
        for elem_type, count in sorted(element_types.items()):
            print(f"  {elem_type}: {count}")

        # Print detected states
        print("\nDetected states:")
        for state in capture.states:
            print(f"  {state.name} ({state.state_type.value})")
            print(f"    Elements: {len(state.element_ids)}")
            print(f"    BBox: {state.bbox.width}x{state.bbox.height}")

        # End session
        completed_session = await extractor.end_session()
        print(f"\nSession completed with {len(completed_session.captures)} captures")

        return completed_session

    finally:
        await extractor.disconnect()

# Run
if __name__ == "__main__":
    session = asyncio.run(basic_web_extraction())
```

---

## Web App with Authentication

Extract from an authenticated web application.

```python
import asyncio
from pathlib import Path
from qontinui.extraction.runtime import (
    PlaywrightExtractor,
    ExtractionTarget,
    RuntimeType,
)

async def authenticated_extraction():
    """Extract from an app requiring authentication."""

    target = ExtractionTarget(
        runtime_type=RuntimeType.WEB,
        url="https://app.example.com/dashboard",
        viewport=(1920, 1080),
        headless=False,  # Show browser for debugging

        # Provide authentication cookies
        auth_cookies={
            "session_id": "abc123def456",
            "auth_token": "Bearer xyz789",
        },
    )

    async with PlaywrightExtractor() as extractor:
        session = await extractor.start_session(target)

        # Extract dashboard
        dashboard = await extractor.extract_current_state()
        print(f"Dashboard: {len(dashboard.elements)} elements")

        # Navigate to different pages
        pages = [
            "/dashboard/profile",
            "/dashboard/settings",
            "/dashboard/analytics",
        ]

        for page_path in pages:
            url = f"https://app.example.com{page_path}"
            await extractor.navigate_to_route(url)

            capture = await extractor.extract_current_state()
            print(f"{page_path}: {len(capture.elements)} elements")

        # Get all captures
        completed = await extractor.end_session()
        print(f"\nTotal captures: {len(completed.captures)}")

        return completed

# Run
if __name__ == "__main__":
    session = asyncio.run(authenticated_extraction())
```

---

## Tauri App Extraction

Extract from a Tauri application with mocked APIs.

```python
import asyncio
from pathlib import Path
from qontinui.extraction.runtime import (
    TauriExtractor,
    ExtractionTarget,
    RuntimeType,
)

async def tauri_app_extraction():
    """Extract from a Tauri application."""

    target = ExtractionTarget(
        runtime_type=RuntimeType.TAURI,
        url="http://localhost:1420",

        # Auto-start dev server
        app_dev_command="npm run tauri dev",

        # Load Tauri config
        tauri_config_path="./src-tauri/tauri.conf.json",

        # Provide custom API mocks
        tauri_mocks={
            # Mock user data
            "get_user": {
                "id": 1,
                "name": "Test User",
                "email": "test@example.com",
                "role": "admin"
            },

            # Mock app settings
            "load_settings": {
                "theme": "dark",
                "language": "en",
                "notifications": True,
                "auto_save": True
            },

            # Mock data fetch
            "fetch_items": [
                {"id": 1, "title": "Item 1", "status": "active"},
                {"id": 2, "title": "Item 2", "status": "pending"},
                {"id": 3, "title": "Item 3", "status": "completed"},
            ],

            # Mock save operation
            "save_item": True,
            "delete_item": True,
        },

        viewport=(1280, 720),
        headless=False,
    )

    async with TauriExtractor() as extractor:
        session = await extractor.start_session(
            target,
            storage_dir=Path("./output/tauri_extraction")
        )

        # Wait for Tauri app to initialize
        await extractor.wait_for_stability(3000)

        # Extract initial state
        initial = await extractor.extract_current_state()
        print(f"Initial state: {len(initial.elements)} elements")

        # Simulate a Tauri event
        await extractor.simulate_tauri_event(
            "notification",
            {"title": "Test", "message": "Hello from mock!"}
        )

        # Add custom mock at runtime
        await extractor.inject_custom_mock(
            "get_stats",
            {"total": 42, "active": 10, "pending": 5}
        )

        # Interact with the app
        await extractor.perform_action("click", "button.refresh")
        await extractor.wait_for_stability(1000)

        refreshed = await extractor.extract_current_state()
        print(f"After refresh: {len(refreshed.elements)} elements")

        # Test different views
        views = [".view-list", ".view-grid", ".view-timeline"]
        for view_selector in views:
            try:
                await extractor.perform_action("click", view_selector)
                capture = await extractor.extract_current_state()
                print(f"View {view_selector}: {len(capture.elements)} elements")
            except Exception as e:
                print(f"View {view_selector} not found: {e}")

        # Get completed session
        completed = await extractor.end_session()
        print(f"\nSession completed:")
        print(f"  Total captures: {len(completed.captures)}")
        print(f"  Storage: {completed.storage_dir}")

        return completed

# Run
if __name__ == "__main__":
    session = asyncio.run(tauri_app_extraction())
```

---

## State Comparison

Compare states before and after an action.

```python
import asyncio
from qontinui.extraction.runtime import (
    PlaywrightExtractor,
    ExtractionTarget,
    RuntimeType,
)

async def compare_states():
    """Compare UI states before and after an action."""

    target = ExtractionTarget(
        runtime_type=RuntimeType.WEB,
        url="https://example.com",
        viewport=(1920, 1080),
    )

    async with PlaywrightExtractor() as extractor:
        await extractor.start_session(target)

        # Capture state before action
        before = await extractor.extract_current_state()

        # Perform action
        after = await extractor.perform_action(
            action_type="click",
            target_selector="button.show-menu"
        )

        # Compare captures
        print("State Comparison:")
        print(f"  Before: {len(before.elements)} elements, {len(before.states)} states")
        print(f"  After:  {len(after.elements)} elements, {len(after.states)} states")

        # Find new elements
        before_ids = {e.id for e in before.elements}
        new_elements = [e for e in after.elements if e.id not in before_ids]

        print(f"\nNew elements appeared: {len(new_elements)}")
        for elem in new_elements:
            print(f"  - {elem.element_type.value}: {elem.text_content}")

        # Find new states
        before_state_ids = {s.id for s in before.states}
        new_states = [s for s in after.states if s.id not in before_state_ids]

        print(f"\nNew states appeared: {len(new_states)}")
        for state in new_states:
            print(f"  - {state.name} ({state.state_type.value})")

# Run
if __name__ == "__main__":
    asyncio.run(compare_states())
```

---

## Interactive Exploration

Interactively explore an application.

```python
import asyncio
from qontinui.extraction.runtime import (
    PlaywrightExtractor,
    ExtractionTarget,
    RuntimeType,
)

async def interactive_exploration():
    """Interactively explore an application."""

    target = ExtractionTarget(
        runtime_type=RuntimeType.WEB,
        url="https://example.com",
        viewport=(1920, 1080),
        headless=False,  # Show browser
    )

    async with PlaywrightExtractor() as extractor:
        session = await extractor.start_session(target)

        print("Interactive Exploration Started")
        print("Commands: extract, click <selector>, type <selector> <text>, nav <url>, quit")

        while True:
            cmd = input("\n> ").strip()

            if cmd == "quit":
                break

            elif cmd == "extract":
                capture = await extractor.extract_current_state()
                print(f"Captured: {len(capture.elements)} elements, {len(capture.states)} states")

                # Show interactive elements
                print("\nInteractive elements:")
                for elem in capture.elements[:10]:  # First 10
                    text = elem.text_content or elem.aria_label or ""
                    print(f"  {elem.element_type.value}: {text[:50]}")
                    print(f"    Selector: {elem.selector}")

            elif cmd.startswith("click "):
                selector = cmd[6:]
                try:
                    await extractor.perform_action("click", selector)
                    print(f"Clicked: {selector}")
                except Exception as e:
                    print(f"Error: {e}")

            elif cmd.startswith("type "):
                parts = cmd[5:].split(" ", 1)
                if len(parts) == 2:
                    selector, text = parts
                    try:
                        await extractor.perform_action("type", selector, text)
                        print(f"Typed '{text}' into {selector}")
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print("Usage: type <selector> <text>")

            elif cmd.startswith("nav "):
                url = cmd[4:]
                try:
                    await extractor.navigate_to_route(url)
                    print(f"Navigated to: {url}")
                except Exception as e:
                    print(f"Error: {e}")

            else:
                print("Unknown command")

        # End session
        completed = await extractor.end_session()
        print(f"\nSession ended with {len(completed.captures)} captures")

# Run
if __name__ == "__main__":
    asyncio.run(interactive_exploration())
```

---

## Advanced: Custom Extractor

Create a custom extractor for a specific framework.

```python
import asyncio
from qontinui.extraction.runtime import (
    PlaywrightExtractor,
    ExtractionTarget,
    RuntimeType,
)

class ReactExtractor(PlaywrightExtractor):
    """Custom extractor for React applications."""

    async def extract_react_components(self):
        """Extract React component tree."""
        if not self.page:
            return []

        components = await self.page.evaluate("""
            () => {
                // Find React Fiber root
                const findReactRoot = (element) => {
                    for (let key in element) {
                        if (key.startsWith('__reactInternalInstance') ||
                            key.startsWith('__reactFiber')) {
                            return element[key];
                        }
                    }
                    return null;
                };

                // Extract component info
                const extractComponent = (fiber) => {
                    if (!fiber) return null;

                    return {
                        type: fiber.type?.name || fiber.elementType?.name || 'Unknown',
                        props: fiber.memoizedProps || {},
                        state: fiber.memoizedState || null,
                    };
                };

                // Get all React components
                const components = [];
                const root = findReactRoot(document.querySelector('#root'));

                if (root) {
                    let fiber = root;
                    while (fiber) {
                        const component = extractComponent(fiber);
                        if (component) {
                            components.push(component);
                        }
                        fiber = fiber.child || fiber.sibling;
                    }
                }

                return components;
            }
        """)

        return components

    async def extract_redux_state(self):
        """Extract Redux store state."""
        if not self.page:
            return None

        state = await self.page.evaluate("""
            () => {
                // Access Redux DevTools extension
                if (window.__REDUX_DEVTOOLS_EXTENSION__) {
                    const store = window.__REDUX_DEVTOOLS_EXTENSION__.store;
                    return store ? store.getState() : null;
                }
                return null;
            }
        """)

        return state

async def react_app_extraction():
    """Extract from a React application."""

    target = ExtractionTarget(
        runtime_type=RuntimeType.WEB,
        url="http://localhost:3000",
        viewport=(1920, 1080),
        headless=False,
    )

    async with ReactExtractor() as extractor:
        await extractor.start_session(target)

        # Extract standard UI
        capture = await extractor.extract_current_state()
        print(f"UI Elements: {len(capture.elements)}")

        # Extract React-specific data
        components = await extractor.extract_react_components()
        print(f"React Components: {len(components)}")

        redux_state = await extractor.extract_redux_state()
        if redux_state:
            print(f"Redux State Keys: {list(redux_state.keys())}")

# Run
if __name__ == "__main__":
    asyncio.run(react_app_extraction())
```

---

## Tips and Best Practices

### 1. Handle Asynchronous Operations

Always use `await` with extractor methods:

```python
# Good
capture = await extractor.extract_current_state()

# Bad
capture = extractor.extract_current_state()  # Returns a coroutine!
```

### 2. Use Context Managers

Use `async with` for automatic cleanup:

```python
# Good
async with PlaywrightExtractor() as extractor:
    await extractor.start_session(target)
    # ... extraction code ...
    # Automatic cleanup on exit

# Also good
extractor = PlaywrightExtractor()
try:
    await extractor.start_session(target)
    # ... extraction code ...
finally:
    await extractor.disconnect()
```

### 3. Wait for Stability

Wait for animations and transitions to complete:

```python
await extractor.perform_action("click", "button.menu")
await extractor.wait_for_stability(1000)  # Wait 1 second
capture = await extractor.extract_current_state()
```

### 4. Handle Errors

Gracefully handle extraction errors:

```python
try:
    capture = await extractor.extract_current_state()
except RuntimeError as e:
    print(f"Extraction failed: {e}")
    # Retry or skip
```

### 5. Storage Management

Organize extraction outputs:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
storage_dir = Path(f"./extractions/{timestamp}")

session = await extractor.start_session(
    target,
    storage_dir=storage_dir
)
```

---

## Running Examples

```bash
# Install dependencies
pip install playwright
playwright install chromium

# Run web extraction
python examples/basic_web_extraction.py

# Run Tauri extraction
cd tauri-app
npm install
cd ..
python examples/tauri_app_extraction.py

# Interactive exploration
python examples/interactive_exploration.py
```
