# Qontinui Web Extraction Examples

Example scripts demonstrating qontinui's web extraction capabilities.

## Prerequisites

```bash
cd qontinui
poetry install
playwright install chromium
```

## Examples

### 1. Web Scraping (`web_scraping_example.py`)

Extract interactive elements (links, buttons, inputs) from web pages.

```bash
# Basic usage
poetry run python scripts/examples/web_scraping_example.py

# Custom URL
poetry run python scripts/examples/web_scraping_example.py --url https://example.com

# Show browser (not headless)
poetry run python scripts/examples/web_scraping_example.py --no-headless

# Save results to JSON
poetry run python scripts/examples/web_scraping_example.py --output results.json

# Run specific mode
poetry run python scripts/examples/web_scraping_example.py --mode basic
poetry run python scripts/examples/web_scraping_example.py --mode enhanced
poetry run python scripts/examples/web_scraping_example.py --mode forms
```

**Features demonstrated:**
- Basic element extraction (buttons, links, inputs)
- Enhanced extraction with shadow DOM support
- LLM-friendly element formatting
- JSON output formatting
- Form element extraction

### 2. Form Filling (`form_filling_example.py`)

Find form fields using natural language descriptions and demonstrate form interactions.

```bash
# Basic usage (uses MockLLMClient, no API key needed)
poetry run python scripts/examples/form_filling_example.py

# Custom URL
poetry run python scripts/examples/form_filling_example.py --url https://example.com/login

# Run specific demo
poetry run python scripts/examples/form_filling_example.py --demo search
poetry run python scripts/examples/form_filling_example.py --demo actions
poetry run python scripts/examples/form_filling_example.py --demo fallback
poetry run python scripts/examples/form_filling_example.py --demo multiple
```

**Features demonstrated:**
- Natural language element selection (`NaturalLanguageSelector`)
- Finding form fields by description ("the email input", "submit button")
- Action selection ("click the sign in button", "type in the search box")
- Fallback text-based selection (no LLM required)
- Finding multiple matching elements

### 3. Element Healing (`element_healing_example.py`)

Demonstrate automatic selector repair when DOM changes.

```bash
# Basic usage
poetry run python scripts/examples/element_healing_example.py

# Run specific demo
poetry run python scripts/examples/element_healing_example.py --demo variations
poetry run python scripts/examples/element_healing_example.py --demo text
poetry run python scripts/examples/element_healing_example.py --demo history
poetry run python scripts/examples/element_healing_example.py --demo overview
```

**Features demonstrated:**
- Selector healing strategies (variations, text match, aria match)
- Healing history (learning from past repairs)
- Strategy statistics and prioritization
- Position-based element recovery

### 4. Accessibility Audit (`accessibility_audit_example.py`)

Extract accessibility tree and analyze pages for common issues.

```bash
# Basic usage
poetry run python scripts/examples/accessibility_audit_example.py

# Save report to JSON
poetry run python scripts/examples/accessibility_audit_example.py --output report.json

# Run specific demo
poetry run python scripts/examples/accessibility_audit_example.py --demo tree
poetry run python scripts/examples/accessibility_audit_example.py --demo enrich
poetry run python scripts/examples/accessibility_audit_example.py --demo audit
```

**Features demonstrated:**
- Accessibility tree extraction
- Element enrichment with a11y data
- Missing label/role detection
- Form input label checking
- Generating accessibility reports

## Using with Real LLM Providers

The examples use `MockLLMClient` by default (no API key required). To use real LLMs:

```python
from qontinui.extraction.web.llm_clients import AnthropicClient, OpenAIClient

# Anthropic Claude (set ANTHROPIC_API_KEY env var)
client = AnthropicClient()

# OpenAI GPT (set OPENAI_API_KEY env var)
client = OpenAIClient()

# Pass to NaturalLanguageSelector
selector = NaturalLanguageSelector(client)
```

## Common Patterns

### Extract Elements from a Page

```python
from playwright.async_api import async_playwright
from qontinui.extraction.web import InteractiveElementExtractor

async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto("https://example.com")

    extractor = InteractiveElementExtractor()
    elements = await extractor.extract_interactive_elements(page, "demo")

    for elem in elements:
        print(f"<{elem.tag_name}> {elem.text} - {elem.selector}")
```

### Find Element by Description

```python
from qontinui.extraction.web import NaturalLanguageSelector
from qontinui.extraction.web.llm_clients import MockLLMClient

client = MockLLMClient()
selector = NaturalLanguageSelector(client)

result = await selector.find_element("the login button", elements)
if result.found:
    print(f"Found: {result.element.selector}")
```

### Heal a Broken Selector

```python
from qontinui.extraction.web import SelectorHealer

healer = SelectorHealer()
result = await healer.heal_selector(
    broken_selector="button.old-class",
    original_element=element,
    page=page,
)
if result.success:
    print(f"Healed to: {result.healed_selector}")
```

### Extract Accessibility Tree

```python
from qontinui.extraction.web import extract_accessibility_tree

tree = await extract_accessibility_tree(page)
buttons = tree.find_by_role("button")
print(f"Found {len(buttons)} buttons")
```
