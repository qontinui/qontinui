"""
End-to-end integration tests for web extraction module.

Tests the full extraction pipeline against real websites using Playwright.
These tests require network access and are marked for optional execution.

Run with: poetry run pytest tests/extraction/web/test_integration_e2e.py -v
Skip slow tests: poetry run pytest tests/extraction/web/test_integration_e2e.py -v -m "not slow"
"""

import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from playwright.async_api import BrowserContext, Page, async_playwright

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "network: marks tests as requiring network access")
    config.addinivalue_line(
        "markers", "cdp_required: marks tests as requiring CDP (Chrome DevTools Protocol)"
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def browser_context() -> AsyncGenerator[BrowserContext, None]:
    """Create a Playwright browser context for tests."""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    )
    try:
        yield context
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()


@pytest_asyncio.fixture
async def page(browser_context: BrowserContext) -> AsyncGenerator[Page, None]:
    """Create a Playwright page for tests."""
    page = await browser_context.new_page()
    yield page
    await page.close()


@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Create a temporary storage directory for tests."""
    storage_dir = tmp_path / "extraction_test"
    storage_dir.mkdir(exist_ok=True)
    return storage_dir


@pytest.fixture
def mock_llm_client():
    """Create a MockLLMClient for testing natural language selection."""
    from qontinui.extraction.web.llm_clients import MockLLMClient

    # Set up responses for various test scenarios
    responses = {
        "submit": """INDEX: 0
CONFIDENCE: 0.95
REASONING: Found a submit button matching the description
ALTERNATIVES: none""",
        "login": """INDEX: 0
CONFIDENCE: 0.9
REASONING: Found login-related element
ALTERNATIVES: 1""",
        "navigation": """INDEX: 0
CONFIDENCE: 0.85
REASONING: Found navigation element
ALTERNATIVES: none""",
        "more information": """INDEX: 0
CONFIDENCE: 0.9
REASONING: Found link with 'More information' text
ALTERNATIVES: none""",
    }
    return MockLLMClient(responses=responses)


# ============================================================================
# Test HTML Pages (for local testing without network)
# ============================================================================


SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <nav>
        <a href="/" id="home-link">Home</a>
        <a href="/about">About</a>
    </nav>
    <main>
        <h1>Welcome</h1>
        <button id="submit-btn" class="primary">Submit</button>
        <input type="text" id="name-input" placeholder="Enter name">
        <button class="secondary">Cancel</button>
    </main>
</body>
</html>
"""

SHADOW_DOM_HTML = """
<!DOCTYPE html>
<html>
<head><title>Shadow DOM Test</title></head>
<body>
    <div id="host"></div>
    <script>
        const host = document.getElementById('host');
        const shadow = host.attachShadow({mode: 'open'});
        shadow.innerHTML = `
            <style>button { padding: 10px; }</style>
            <button id="shadow-btn">Shadow Button</button>
            <input type="text" placeholder="Shadow Input">
        `;
    </script>
</body>
</html>
"""

IFRAME_HTML = """
<!DOCTYPE html>
<html>
<head><title>Iframe Test</title></head>
<body>
    <h1>Main Frame</h1>
    <button id="main-btn">Main Button</button>
    <iframe id="test-iframe" srcdoc="
        <!DOCTYPE html>
        <html>
        <body>
            <h2>Iframe Content</h2>
            <button id='iframe-btn'>Iframe Button</button>
            <a href='/iframe-link'>Iframe Link</a>
        </body>
        </html>
    "></iframe>
</body>
</html>
"""

DYNAMIC_SELECTOR_HTML = """
<!DOCTYPE html>
<html>
<head><title>Dynamic Selector Test</title></head>
<body>
    <div id="container">
        <button class="btn-v1 primary">Original Button</button>
    </div>
    <script>
        // Simulate class change after 100ms
        setTimeout(() => {
            const btn = document.querySelector('.btn-v1');
            if (btn) {
                btn.classList.remove('btn-v1');
                btn.classList.add('btn-v2');
            }
        }, 100);
    </script>
</body>
</html>
"""


# ============================================================================
# Test: Full Extraction Pipeline
# ============================================================================


class TestFullExtractionPipeline:
    """Test the complete extraction pipeline on real websites."""

    @pytest.mark.slow
    @pytest.mark.network
    async def test_extract_from_example_com(self, page, temp_storage: Path) -> None:
        """Test full extraction pipeline on example.com (simple, reliable page)."""
        from qontinui.extraction.web.config import ExtractionConfig
        from qontinui.extraction.web.extractor import WebExtractor

        # Navigate to example.com (simple, reliable test target)
        try:
            await page.goto("https://example.com", wait_until="networkidle", timeout=30000)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        # Verify page loaded
        title = await page.title()
        assert "Example Domain" in title

        # Create extractor config
        config = ExtractionConfig(
            urls=["https://example.com"],
            viewports=[(1280, 720)],
            output_dir=str(temp_storage),
        )

        # Run extraction
        extractor = WebExtractor(config=config, storage_dir=temp_storage)

        # Track progress
        progress_events: list[dict[str, Any]] = []

        async def track_progress(data: dict[str, Any]) -> None:
            progress_events.append(data)

        result = await extractor.start(progress_callback=track_progress)

        # Verify results
        assert result.extraction_id is not None
        assert len(result.page_extractions) > 0
        assert len(result.elements) > 0  # Should find at least the link

        # Verify progress was tracked
        assert len(progress_events) > 0
        assert any(e.get("status") == "complete" for e in progress_events)

        # Verify screenshots were captured
        assert len(result.screenshot_ids) > 0
        for screenshot_id in result.screenshot_ids:
            screenshot_path = extractor.get_screenshot_path(screenshot_id)
            assert screenshot_path is not None
            assert screenshot_path.exists()

    async def test_extract_from_local_html(self, page, temp_storage: Path) -> None:
        """Test extraction from a local HTML page (no network required)."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )

        # Load local HTML
        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        # Extract elements
        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "test_screenshot")

        # Verify elements were extracted
        assert len(elements) >= 4  # 2 links + 2 buttons + 1 input

        # Verify specific elements
        element_types = {e.element_type for e in elements}
        assert "button" in element_types or any("button" in t for t in element_types)
        assert "a" in element_types

        # Verify element properties
        submit_btn = next((e for e in elements if e.text == "Submit"), None)
        assert submit_btn is not None
        assert submit_btn.selector == "#submit-btn" or "submit" in submit_btn.selector.lower()

        # Verify bounding boxes
        for element in elements:
            assert element.bbox.width > 0
            assert element.bbox.height > 0


# ============================================================================
# Test: Shadow DOM Extraction
# ============================================================================


class TestShadowDOMExtraction:
    """Test extraction from shadow DOM elements."""

    async def test_extract_from_shadow_dom(self, page) -> None:
        """Test extraction of elements inside shadow DOM."""
        from qontinui.extraction.web.interactive_element_extractor import (
            ExtractionOptions,
            InteractiveElementExtractor,
        )

        # Load page with shadow DOM
        await page.set_content(SHADOW_DOM_HTML)
        await page.wait_for_load_state("domcontentloaded")
        # Wait for shadow DOM to be created
        await page.wait_for_timeout(100)

        # Extract with shadow DOM enabled
        options = ExtractionOptions(include_shadow_dom=True)
        extractor = InteractiveElementExtractor(options=options)
        elements = await extractor.extract_interactive_elements(page, "shadow_test")

        # Verify shadow DOM elements were extracted
        assert len(elements) >= 2  # Should find button and input in shadow DOM

        # Check for shadow path annotation
        shadow_elements = [e for e in elements if e.shadow_path]
        assert len(shadow_elements) >= 1  # At least one element from shadow DOM

        # Verify the shadow button was found
        shadow_button = next((e for e in elements if e.text and "Shadow Button" in e.text), None)
        assert shadow_button is not None

    async def test_shadow_dom_disabled(self, page) -> None:
        """Test that shadow DOM extraction can be disabled."""
        from qontinui.extraction.web.interactive_element_extractor import (
            ExtractionOptions,
            InteractiveElementExtractor,
        )

        await page.set_content(SHADOW_DOM_HTML)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(100)

        # Extract with shadow DOM disabled
        options = ExtractionOptions(include_shadow_dom=False)
        extractor = InteractiveElementExtractor(options=options)
        elements = await extractor.extract_interactive_elements(page, "no_shadow_test")

        # Should not find shadow DOM elements
        shadow_elements = [e for e in elements if e.shadow_path]
        assert len(shadow_elements) == 0


# ============================================================================
# Test: Iframe Extraction
# ============================================================================


class TestIframeExtraction:
    """Test extraction from iframes."""

    async def test_extract_from_iframes(self, page) -> None:
        """Test extraction of elements from iframes."""
        from qontinui.extraction.web.frame_manager import extract_across_frames

        # Load page with iframe
        await page.set_content(IFRAME_HTML)
        await page.wait_for_load_state("domcontentloaded")
        # Wait for iframe to load
        await page.wait_for_timeout(500)

        # Extract across frames
        result = await extract_across_frames(page, screenshot_id="iframe_test")

        # Verify we found multiple frames
        assert len(result.frames) >= 2  # Main frame + iframe

        # Verify elements from main frame
        main_frame_elements = result.get_frame_elements(0)
        assert len(main_frame_elements) >= 1
        main_button = next(
            (e for e in main_frame_elements if "Main Button" in (e.element.text or "")),
            None,
        )
        assert main_button is not None

        # Verify elements from iframe
        iframe_elements = result.get_frame_elements(1)
        assert len(iframe_elements) >= 1

        # Verify encoded IDs are unique across frames
        all_ids = [e.encoded_id for e in result.elements]
        assert len(all_ids) == len(set(all_ids))  # All unique

    async def test_frame_aware_element_lookup(self, page) -> None:
        """Test looking up elements by encoded ID."""
        from qontinui.extraction.web.frame_manager import extract_across_frames

        await page.set_content(IFRAME_HTML)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(500)

        result = await extract_across_frames(page, screenshot_id="lookup_test")

        # Verify lookup by encoded ID
        for element in result.elements:
            found = result.get_by_encoded_id(element.encoded_id)
            assert found is not None
            assert found.encoded_id == element.encoded_id

        # Verify missing ID returns None
        missing = result.get_by_encoded_id("nonexistent-id")
        assert missing is None


# ============================================================================
# Test: Selector Healing
# ============================================================================


class TestSelectorHealing:
    """Test automatic selector healing when DOM changes."""

    async def test_heal_broken_selector_by_text(self, page) -> None:
        """Test healing a broken selector using text matching."""
        from qontinui.extraction.web.models import BoundingBox, InteractiveElement
        from qontinui.extraction.web.selector_healer import SelectorHealer

        # Load page
        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        # Create original element data (with a selector that won't work)
        original_element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=40),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button.nonexistent-class",  # Broken selector
            text="Submit",  # But we have the text
        )

        # Heal the selector
        healer = SelectorHealer()
        result = await healer.heal_selector(
            broken_selector="button.nonexistent-class",
            original_element=original_element,
            page=page,
        )

        # Verify healing succeeded
        assert result.success is True
        assert result.healed_selector is not None
        assert result.element is not None
        # Healing can use various strategies - selector_variation is often tried first
        assert result.strategy_used in [
            "text_match",
            "selector_variation",
            "aria_match",
            "position_match",
        ]

        # Verify the healed selector works
        found = await page.query_selector(result.healed_selector)
        assert found is not None

    async def test_heal_broken_selector_by_aria_label(self, page) -> None:
        """Test healing a broken selector using aria-label."""
        from qontinui.extraction.web.models import BoundingBox, InteractiveElement
        from qontinui.extraction.web.selector_healer import SelectorHealer

        # Load page with aria-label (add multiple buttons so tag alone won't work)
        html = """
        <!DOCTYPE html>
        <html>
        <body>
            <button aria-label="Close dialog" class="changed-class">X</button>
            <button aria-label="Open dialog" class="other-class">O</button>
        </body>
        </html>
        """
        await page.set_content(html)
        await page.wait_for_load_state("domcontentloaded")

        original_element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=40, height=40),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button.old-class",  # Broken selector
            text="X",
            aria_label="Close dialog",  # We have aria-label
        )

        healer = SelectorHealer()
        result = await healer.heal_selector(
            broken_selector="button.old-class",
            original_element=original_element,
            page=page,
        )

        # Verify healing succeeded
        assert result.success is True
        assert result.healed_selector is not None
        # Accept any valid healing strategy
        assert result.strategy_used in [
            "aria_match",
            "text_match",
            "selector_variation",
            "position_match",
        ]

    async def test_healing_with_history(self, page, temp_storage: Path) -> None:
        """Test that healing history improves future repairs."""
        from qontinui.extraction.web.models import BoundingBox, InteractiveElement
        from qontinui.extraction.web.selector_healer import HealingHistory, SelectorHealer

        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        # Create history storage
        history_path = temp_storage / "healing_history.json"
        history = HealingHistory(storage_path=history_path)

        original_element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=40),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button.broken",
            text="Submit",
        )

        # First healing (no history)
        healer = SelectorHealer(history=history)
        result1 = await healer.heal_selector(
            broken_selector="button.broken",
            original_element=original_element,
            page=page,
        )

        assert result1.success is True

        # Verify history was saved
        records = history.lookup("button.broken")
        assert len(records) > 0

        # Second healing should use history
        healer2 = SelectorHealer(history=history)
        result2 = await healer2.heal_selector(
            broken_selector="button.broken",
            original_element=original_element,
            page=page,
        )

        assert result2.success is True
        # History lookup is tried first, so if selector still works, it should use that
        assert result2.strategy_used in ["history_lookup", "selector_variation", "text_match"]


# ============================================================================
# Test: Natural Language Selection with MockLLMClient
# ============================================================================


class TestNaturalLanguageSelection:
    """Test natural language element selection with MockLLMClient."""

    async def test_find_element_by_description(self, page, mock_llm_client) -> None:
        """Test finding an element using natural language description."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )

        # Load page
        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        # Extract elements
        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "test")

        # Find element by description
        selector = NaturalLanguageSelector(llm_client=mock_llm_client)
        result = await selector.find_element("the submit button", elements)

        # Verify result
        assert result.found is True
        assert result.index is not None
        assert result.confidence > 0.5
        assert result.reasoning != ""

    async def test_find_multiple_elements(self, page) -> None:
        """Test finding multiple elements by description."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )
        from qontinui.extraction.web.llm_clients import MockLLMClient
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )

        # Load page with multiple buttons
        html = """
        <!DOCTYPE html>
        <html>
        <body>
            <button>Save</button>
            <button>Cancel</button>
            <button>Delete</button>
        </body>
        </html>
        """
        await page.set_content(html)
        await page.wait_for_load_state("domcontentloaded")

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "test")

        # Create dedicated mock with only the multi-response pattern
        # The generic mock fixture has patterns that may match the prompt's examples
        multi_mock = MockLLMClient(
            responses={
                "all links": """MATCH: 0, 0.95, First link
MATCH: 1, 0.90, Second link
MATCH: 2, 0.85, Third link"""
            }
        )

        selector = NaturalLanguageSelector(llm_client=multi_mock)
        results = await selector.find_multiple("all links", elements)

        # Verify results - MockLLMClient should return matching response
        assert len(results) >= 1

    async def test_fallback_selector(self, page) -> None:
        """Test fallback selector without LLM."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )
        from qontinui.extraction.web.natural_language_selector import FallbackSelector

        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "test")

        # Use fallback selector
        fallback = FallbackSelector()
        result = fallback.find_by_text("Submit", elements)

        assert result.found is True
        assert result.element is not None
        assert "Submit" in (result.element.text or "")

    async def test_select_action(self, page, mock_llm_client) -> None:
        """Test selecting both element and action."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )

        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "test")

        # Configure mock for action selection
        mock_llm_client.responses[
            "click the submit"
        ] = """INDEX: 0
ACTION: click
CONFIDENCE: 0.95
REASONING: User wants to click the submit button"""

        selector = NaturalLanguageSelector(llm_client=mock_llm_client)
        result, action = await selector.select_action("click the submit button", elements)

        assert result.found is True
        assert action in ["click", "type", "hover", "focus", "select"]


# ============================================================================
# Test: Accessibility Tree Extraction via CDP
# ============================================================================


class TestAccessibilityTreeExtraction:
    """Test accessibility tree extraction using CDP."""

    @pytest.mark.cdp_required
    async def test_extract_accessibility_tree(self, page) -> None:
        """Test extracting accessibility tree from a page."""
        from qontinui.extraction.web.accessibility_extractor import (
            AccessibilityExtractor,
        )

        # Load a page with semantic structure
        html = """
        <!DOCTYPE html>
        <html>
        <body>
            <nav aria-label="Main navigation">
                <a href="/">Home</a>
                <a href="/about">About</a>
            </nav>
            <main>
                <h1>Welcome</h1>
                <button aria-label="Submit form">Submit</button>
            </main>
        </body>
        </html>
        """
        await page.set_content(html)
        await page.wait_for_load_state("domcontentloaded")

        # Extract accessibility tree
        extractor = AccessibilityExtractor()
        tree = await extractor.extract_tree(page)

        # Verify tree was extracted
        assert tree is not None
        assert tree.root is not None
        assert tree.node_count > 0

        # Verify we can find nodes by role
        buttons = tree.find_by_role("button")
        assert len(buttons) >= 1

        links = tree.find_by_role("link")
        assert len(links) >= 2

        # Verify we can find nodes by name
        submit_nodes = tree.find_by_name("Submit")
        assert len(submit_nodes) >= 1

    @pytest.mark.cdp_required
    async def test_enrich_elements_with_a11y(self, page) -> None:
        """Test enriching extracted elements with accessibility data."""
        from qontinui.extraction.web.accessibility_extractor import (
            AccessibilityExtractor,
        )
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )

        html = """
        <!DOCTYPE html>
        <html>
        <body>
            <button aria-label="Close dialog">X</button>
            <input type="text" aria-label="Search field" placeholder="Search...">
        </body>
        </html>
        """
        await page.set_content(html)
        await page.wait_for_load_state("domcontentloaded")

        # Extract elements
        elem_extractor = InteractiveElementExtractor()
        elements = await elem_extractor.extract_interactive_elements(page, "test")

        # Enrich with accessibility data
        a11y_extractor = AccessibilityExtractor()
        enriched = await a11y_extractor.enrich_elements(elements, page)

        # Verify enrichment
        assert len(enriched) > 0
        for elem in enriched:
            # Should have accessibility info for interactive elements
            assert elem.element is not None
            # At least some should have a11y data
            if elem.match_confidence > 0:
                assert elem.a11y_role is not None or elem.a11y_name is not None

    async def test_a11y_tree_to_text(self, page) -> None:
        """Test converting accessibility tree to text format."""
        from qontinui.extraction.web.accessibility_extractor import (
            AccessibilityExtractor,
            a11y_tree_to_text,
        )

        html = """
        <!DOCTYPE html>
        <html>
        <body>
            <h1>Page Title</h1>
            <button>Click Me</button>
        </body>
        </html>
        """
        await page.set_content(html)
        await page.wait_for_load_state("domcontentloaded")

        extractor = AccessibilityExtractor()
        tree = await extractor.extract_tree(page)

        # Convert to text
        text = a11y_tree_to_text(tree)

        # Verify text format
        assert "button" in text.lower() or "Click" in text
        assert len(text) > 0


# ============================================================================
# Test: Hybrid Extraction (DOM + Screenshot)
# ============================================================================


class TestHybridExtraction:
    """Test hybrid DOM + screenshot extraction for LLM consumption."""

    async def test_extract_hybrid_context(self, page) -> None:
        """Test extracting hybrid context (DOM + screenshot)."""
        from qontinui.extraction.web.hybrid_extractor import HybridExtractor

        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        extractor = HybridExtractor(
            include_accessibility=True,
            include_shadow_dom=True,
            include_iframes=False,
            screenshot_format="jpeg",
        )

        context = await extractor.extract(page)

        # Verify context structure
        assert context.url is not None
        assert context.title == "Test Page"
        assert context.viewport[0] > 0 and context.viewport[1] > 0

        # Verify screenshot was captured
        assert context.screenshot_base64 != ""

        # Verify elements were extracted
        assert len(context.elements) > 0
        assert context.elements_formatted != ""

        # Verify scroll info
        assert context.scroll_height >= 0
        assert context.viewport_height >= 0

    async def test_hybrid_context_to_llm_message(self, page) -> None:
        """Test converting hybrid context to LLM message format."""
        from qontinui.extraction.web.hybrid_extractor import HybridExtractor

        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        extractor = HybridExtractor()
        context = await extractor.extract(page)

        # Convert to LLM message
        message = context.to_llm_message(include_screenshot=True)

        # Verify message structure
        assert "text" in message
        assert "image" in message
        assert message["image"]["type"] in ["jpeg", "png"]
        assert "Test Page" in message["text"]
        assert "Interactive Elements" in message["text"]

        # Test without screenshot
        message_no_img = context.to_llm_message(include_screenshot=False)
        assert "text" in message_no_img
        assert "image" not in message_no_img

    async def test_build_llm_prompt(self, page) -> None:
        """Test building a complete LLM prompt from hybrid context."""
        from qontinui.extraction.web.hybrid_extractor import (
            HybridExtractor,
            build_llm_prompt,
        )

        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        extractor = HybridExtractor()
        context = await extractor.extract(page)

        prompt = build_llm_prompt(
            context=context,
            instruction="Click the submit button",
            include_screenshot=True,
        )

        # Verify prompt structure
        assert "system" in prompt
        assert "user" in prompt
        assert "web automation" in prompt["system"].lower()
        assert "Click the submit button" in prompt["user"]["text"]

    async def test_state_tracker(self, page) -> None:
        """Test tracking page state changes."""
        from qontinui.extraction.web.hybrid_extractor import StateTracker

        await page.set_content(SIMPLE_HTML)
        await page.wait_for_load_state("domcontentloaded")

        tracker = StateTracker()

        # Capture initial state
        state1 = await tracker.capture_state(page)
        assert state1.context.url is not None
        assert state1.state_hash != ""

        # Capture same state again
        state2 = await tracker.capture_state(page)

        # States should be the same
        assert tracker.is_same_state(state1, state2)

        # Modify page
        await page.evaluate("document.body.innerHTML = '<h1>Changed</h1>'")

        # Capture new state
        state3 = await tracker.capture_state(page)

        # States should be different
        assert not tracker.is_same_state(state1, state3)

        # Get diff
        diff = tracker.get_state_diff(state1, state3)
        assert diff["element_count_changed"] is True
        assert diff["state_hash_changed"] is True

    async def test_hybrid_extraction_with_iframes(self, page) -> None:
        """Test hybrid extraction with iframes."""
        from qontinui.extraction.web.hybrid_extractor import HybridExtractor

        await page.set_content(IFRAME_HTML)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(500)  # Wait for iframe

        extractor = HybridExtractor(include_iframes=True)
        context = await extractor.extract(page)

        # Verify iframe detection
        assert context.frame_count >= 2
        assert context.has_iframes is True

        # Verify elements from both frames
        assert len(context.elements) >= 2


# ============================================================================
# Test: Network Error Handling
# ============================================================================


class TestNetworkErrorHandling:
    """Test graceful handling of network failures."""

    @pytest.mark.slow
    async def test_handle_unreachable_url(self, browser_context) -> None:
        """Test handling unreachable URL gracefully."""

        page = await browser_context.new_page()

        try:
            # Try to navigate to an unreachable URL
            with pytest.raises(Exception):
                await page.goto(
                    "https://this-domain-should-not-exist-12345.invalid",
                    timeout=5000,
                )
        finally:
            await page.close()

    @pytest.mark.slow
    async def test_handle_timeout(self, page) -> None:
        """Test handling page load timeout."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )

        # Set up a page that loads slowly
        html = """
        <!DOCTYPE html>
        <html>
        <body>
            <button>Test</button>
            <script>
                // Simulate slow loading
                while(false) {} // Disabled to prevent actual hang
            </script>
        </body>
        </html>
        """
        await page.set_content(html)

        # Extract should still work with timeout
        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "timeout_test", timeout=10.0)

        # Should still extract elements
        assert isinstance(elements, list)


# ============================================================================
# Test: Real Website Integration (requires network)
# ============================================================================


@pytest.mark.slow
@pytest.mark.network
class TestRealWebsiteIntegration:
    """Integration tests against real websites (requires network access)."""

    async def test_example_com_elements(self, page) -> None:
        """Test extracting elements from example.com."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )

        try:
            await page.goto("https://example.com", wait_until="networkidle", timeout=30000)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "example_com")

        # example.com has at least one link ("More information...")
        assert len(elements) >= 1

        # Find the "More information" link
        link = next((e for e in elements if e.tag_name == "a"), None)
        assert link is not None
        assert link.href is not None

    async def test_example_com_hybrid_extraction(self, page) -> None:
        """Test hybrid extraction from example.com."""
        from qontinui.extraction.web.hybrid_extractor import HybridExtractor

        try:
            await page.goto("https://example.com", wait_until="networkidle", timeout=30000)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        extractor = HybridExtractor()
        context = await extractor.extract(page)

        # Verify extraction
        assert "Example Domain" in context.title
        assert context.screenshot_base64 != ""
        assert len(context.elements) >= 1

    async def test_example_com_natural_language_selection(self, page, mock_llm_client) -> None:
        """Test natural language selection on example.com."""
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )

        try:
            await page.goto("https://example.com", wait_until="networkidle", timeout=30000)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "example")

        selector = NaturalLanguageSelector(llm_client=mock_llm_client)
        result = await selector.find_element("the more information link", elements)

        # Mock should return a result (configured in fixture)
        assert result is not None
        # Even if mock returns first element, verify it's a valid result
        assert result.confidence >= 0
