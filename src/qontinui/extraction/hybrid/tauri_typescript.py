"""
Tauri + TypeScript hybrid extractor.

Combines static TypeScript analysis with runtime Playwright extraction
to produce States, StateImages, and Transitions for Tauri applications.
"""

import asyncio
import hashlib
import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from qontinui_schemas.common import utc_now

from .base import (
    BoundingBox,
    HybridExtractionConfig,
    HybridExtractionResult,
    ImagePattern,
    State,
    StateImage,
    StateTransition,
    StateType,
    TechStackExtractor,
    TransitionTrigger,
)

logger = logging.getLogger(__name__)


class TauriTypeScriptExtractor(TechStackExtractor):
    """
    Hybrid extractor for Tauri applications with TypeScript frontends.

    This extractor:
    1. Parses TypeScript/TSX files to identify components, state, and event handlers
    2. Reads Tauri configuration for app settings
    3. Starts the Tauri dev server
    4. Uses Playwright to capture screenshots and bounding boxes
    5. Correlates static analysis with runtime data to produce States/StateImages
    """

    tech_stack_name = "tauri-typescript"

    def __init__(self) -> None:
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.dev_process: subprocess.Popen | None = None
        self._screenshot_counter = 0

    @classmethod
    def can_handle(cls, project_path: Path) -> bool:
        """Check if this is a Tauri + TypeScript project."""
        # Check for Tauri config
        tauri_config = project_path / "src-tauri" / "tauri.conf.json"
        if not tauri_config.exists():
            return False

        # Check for TypeScript/JavaScript frontend
        package_json = project_path / "package.json"
        if not package_json.exists():
            return False

        # Could also check for tsconfig.json or .tsx files
        return True

    @classmethod
    def detect_config(cls, project_path: Path) -> HybridExtractionConfig | None:
        """Auto-detect configuration from Tauri project."""
        tauri_config_path = project_path / "src-tauri" / "tauri.conf.json"
        package_json_path = project_path / "package.json"

        if not tauri_config_path.exists() or not package_json_path.exists():
            return None

        try:
            # Read Tauri config
            with open(tauri_config_path, encoding="utf-8") as f:
                tauri_config = json.load(f)

            # Read package.json
            with open(package_json_path, encoding="utf-8") as f:
                package_json = json.load(f)

            # Determine dev command
            scripts = package_json.get("scripts", {})
            dev_command = None
            if "tauri:dev" in scripts:
                dev_command = "npm run tauri:dev"
            elif "tauri dev" in scripts.get("dev", ""):
                dev_command = "npm run dev"
            elif "dev" in scripts:
                dev_command = "npm run dev"

            # Get dev URL from Tauri config
            build_config = tauri_config.get("build", {})
            dev_url = build_config.get("devPath", "http://localhost:3000")
            if dev_url.startswith("http"):
                pass  # It's a URL
            else:
                # It's a path, assume default dev server
                dev_url = "http://localhost:3000"

            # Detect frontend framework
            dependencies = package_json.get("dependencies", {})
            dev_dependencies = package_json.get("devDependencies", {})
            all_deps = {**dependencies, **dev_dependencies}

            framework_hints = {}
            if "react" in all_deps:
                framework_hints["frontend"] = "react"
            elif "vue" in all_deps:
                framework_hints["frontend"] = "vue"
            elif "svelte" in all_deps:
                framework_hints["frontend"] = "svelte"
            elif "solid-js" in all_deps:
                framework_hints["frontend"] = "solid"

            # Check for common frontend paths
            frontend_path = None
            possible_frontend_paths = [
                project_path / "src",
                project_path / "frontend",
                project_path / "app",
            ]
            for path in possible_frontend_paths:
                if path.exists():
                    frontend_path = path
                    break

            return HybridExtractionConfig(
                project_path=project_path,
                frontend_path=frontend_path,
                dev_command=dev_command,
                dev_url=dev_url,
                framework_hints=framework_hints,
            )

        except Exception as e:
            logger.warning(f"Failed to auto-detect config: {e}")
            return None

    async def extract(self, config: HybridExtractionConfig) -> HybridExtractionResult:
        """
        Perform hybrid extraction for Tauri + TypeScript app.

        1. Run static analysis on TypeScript files
        2. Start dev server
        3. Connect via Playwright
        4. Navigate through app capturing states
        5. Extract bounding boxes for interactive elements
        6. Correlate and produce final result
        """
        extraction_id = str(uuid.uuid4())[:8]
        result = HybridExtractionResult(
            extraction_id=extraction_id,
            tech_stack=self.tech_stack_name,
        )

        # Setup output directory
        if config.output_dir:
            output_dir = config.output_dir
        else:
            output_dir = config.project_path / ".qontinui" / "extraction" / extraction_id
        output_dir.mkdir(parents=True, exist_ok=True)
        result.screenshots_dir = output_dir / "screenshots"
        result.screenshots_dir.mkdir(exist_ok=True)

        try:
            # Phase 1: Static Analysis
            logger.info("Phase 1: Running static analysis...")
            static_data = await self._run_static_analysis(config)

            # Phase 2: Start dev server
            logger.info("Phase 2: Starting dev server...")
            await self._start_dev_server(config)

            # Phase 3: Connect Playwright
            logger.info("Phase 3: Connecting via Playwright...")
            await self._connect_playwright(config)

            # Phase 4: Extract states and StateImages
            logger.info("Phase 4: Extracting states and StateImages...")
            states, state_images = await self._extract_states_and_images(
                config, static_data, result.screenshots_dir
            )
            result.states = states
            result.state_images = state_images

            # Phase 5: Extract transitions
            logger.info("Phase 5: Extracting transitions...")
            transitions = await self._extract_transitions(config, static_data, states)
            result.transitions = transitions

            result.completed_at = utc_now()
            logger.info(
                f"Extraction complete: {len(states)} states, "
                f"{len(state_images)} images, {len(transitions)} transitions"
            )

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            result.errors.append(str(e))

        finally:
            await self._cleanup()

        return result

    async def _run_static_analysis(self, config: HybridExtractionConfig) -> dict[str, Any]:
        """Run static analysis on TypeScript files."""
        from ..static.typescript.parser import TypeScriptParser

        static_data: dict[str, Any] = {
            "components": [],
            "state_variables": [],
            "event_handlers": [],
            "conditional_renders": [],
            "navigation_links": [],
            "jsx_elements": [],
        }

        try:
            parser = TypeScriptParser()
            frontend_path = config.frontend_path or config.project_path / "src"

            if frontend_path.exists():
                parse_result = await parser.parse_directory(
                    frontend_path,
                    patterns=["*.tsx", "*.ts", "*.jsx", "*.js"],
                    exclude=[
                        "node_modules/**",
                        "dist/**",
                        "*.test.*",
                        "*.spec.*",
                    ],
                )

                # Aggregate results from all files
                for file_path, file_result in parse_result.files.items():
                    for comp in file_result.components:
                        static_data["components"].append(
                            {
                                "name": comp.name,
                                "type": comp.type,
                                "line": comp.line,
                                "file": file_path,
                                "props": [
                                    {"name": p.name, "default": p.default} for p in comp.props
                                ],
                                "children": comp.children,
                            }
                        )

                    for state_var in file_result.state_variables:
                        static_data["state_variables"].append(
                            {
                                "name": state_var.name,
                                "hook": state_var.hook,
                                "line": state_var.line,
                                "file": file_path,
                                "initial_value": state_var.initial_value,
                                "setter": state_var.setter,
                            }
                        )

                    for handler in file_result.event_handlers:
                        static_data["event_handlers"].append(
                            {
                                "event": handler.event,
                                "name": handler.name,
                                "line": handler.line,
                                "file": file_path,
                                "state_changes": handler.state_changes,
                            }
                        )

                    for cond in file_result.conditional_renders:
                        static_data["conditional_renders"].append(
                            {
                                "condition": cond.condition,
                                "line": cond.line,
                                "file": file_path,
                                "pattern": cond.pattern,
                                "renders_true": cond.renders_true,
                                "renders_false": cond.renders_false,
                            }
                        )

                    for link in file_result.navigation_links:
                        static_data["navigation_links"].append(
                            {
                                "type": link.type,
                                "target": link.target,
                                "line": link.line,
                                "file": file_path,
                                "component": link.component,
                            }
                        )

                    for jsx in file_result.jsx_elements:
                        static_data["jsx_elements"].append(
                            {
                                "name": jsx.name,
                                "line": jsx.line,
                                "file": file_path,
                                "props": jsx.props,
                                "self_closing": jsx.self_closing,
                            }
                        )

                logger.info(
                    f"Static analysis found: {len(static_data['components'])} components, "
                    f"{len(static_data['state_variables'])} state variables, "
                    f"{len(static_data['event_handlers'])} event handlers"
                )

        except Exception as e:
            logger.warning(f"Static analysis error: {e}")
            # Continue with empty static data

        return static_data

    async def _start_dev_server(self, config: HybridExtractionConfig) -> None:
        """Start the Tauri dev server."""
        if not config.dev_command:
            logger.warning("No dev command specified, assuming server is running")
            return

        try:
            # Start the dev server process
            cmd_parts = config.dev_command.split()
            self.dev_process = subprocess.Popen(
                cmd_parts,
                cwd=str(config.project_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for server to be ready
            await self._wait_for_server(config.dev_url)
            logger.info(f"Dev server started at {config.dev_url}")

        except Exception as e:
            logger.error(f"Failed to start dev server: {e}")
            raise

    async def _wait_for_server(self, url: str, timeout: int = 60) -> None:
        """Wait for the dev server to be ready."""
        import aiohttp

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status < 500:
                            return
            except Exception:
                pass

            await asyncio.sleep(1)

        raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")

    async def _connect_playwright(self, config: HybridExtractionConfig) -> None:
        """Connect to the app via Playwright."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=config.headless,
        )
        self.context = await self.browser.new_context(
            viewport={"width": config.viewport[0], "height": config.viewport[1]},
        )
        self.page = await self.context.new_page()

        # Inject Tauri mocks
        await self._inject_tauri_mocks()

        # Navigate to the app
        await self.page.goto(config.dev_url, wait_until="networkidle")
        await self.page.wait_for_timeout(1000)

        logger.info("Connected to app via Playwright")

    async def _inject_tauri_mocks(self) -> None:
        """Inject Tauri API mocks into the page."""
        if not self.page:
            return

        mock_script = """
        window.__TAURI__ = {
            invoke: async (cmd, args) => {
                console.log('[Tauri Mock] invoke:', cmd, args);
                return null;
            },
            event: {
                listen: async (event, handler) => {
                    console.log('[Tauri Mock] listen:', event);
                    return () => {};
                },
                emit: async (event, payload) => {
                    console.log('[Tauri Mock] emit:', event, payload);
                },
                once: async (event, handler) => {
                    console.log('[Tauri Mock] once:', event);
                    return () => {};
                }
            }
        };
        """
        await self.page.add_init_script(mock_script)

    async def _extract_states_and_images(
        self,
        config: HybridExtractionConfig,
        static_data: dict[str, Any],
        screenshots_dir: Path,
    ) -> tuple[list[State], list[StateImage]]:
        """Extract states and StateImages by combining static + runtime data."""
        states: list[State] = []
        state_images: list[StateImage] = []

        if not self.page:
            return states, state_images

        # Create main page state
        main_state = await self._capture_current_state("main", StateType.PAGE, screenshots_dir)
        states.append(main_state)

        # Extract StateImages for interactive elements
        interactive_selectors = await self._find_interactive_selectors()

        for selector_info in interactive_selectors:
            selector = selector_info["selector"]
            try:
                images = await self.extract_state_images_for_selector(
                    selector,
                    capture_hover=config.capture_hover_states,
                    capture_focus=config.capture_focus_states,
                )

                for img in images:
                    # Enrich with static data
                    img = self._enrich_state_image_with_static(img, static_data)
                    state_images.append(img)
                    main_state.state_images.append(img)

            except Exception as e:
                logger.warning(f"Failed to extract StateImage for {selector}: {e}")

        # Look for modal/popup triggers and capture those states
        modal_triggers = await self._find_modal_triggers()
        for trigger in modal_triggers:
            try:
                # Click to open modal
                await self.page.click(trigger["selector"])
                await self.page.wait_for_timeout(500)

                # Capture modal state
                modal_state = await self._capture_current_state(
                    f"modal_{trigger['name']}", StateType.MODAL, screenshots_dir
                )
                modal_state.parent_state_id = main_state.id
                main_state.child_state_ids.append(modal_state.id)
                states.append(modal_state)

                # Extract StateImages in modal
                modal_images = await self._extract_visible_state_images(screenshots_dir)
                for img in modal_images:
                    img = self._enrich_state_image_with_static(img, static_data)
                    state_images.append(img)
                    modal_state.state_images.append(img)

                # Close modal (try common patterns)
                await self._try_close_modal()

            except Exception as e:
                logger.warning(f"Failed to capture modal state: {e}")

        return states, state_images

    async def _capture_current_state(
        self,
        name: str,
        state_type: StateType,
        screenshots_dir: Path,
    ) -> State:
        """Capture the current screen as a State."""
        if not self.page:
            raise RuntimeError("Not connected to page")

        state_id = f"state_{name}_{uuid.uuid4().hex[:6]}"
        self._screenshot_counter += 1
        screenshot_path = screenshots_dir / f"{self._screenshot_counter:04d}_{name}.png"

        await self.page.screenshot(path=str(screenshot_path))

        viewport_size = self.page.viewport_size
        viewport = (
            (viewport_size["width"], viewport_size["height"]) if viewport_size else (1920, 1080)
        )

        return State(
            id=state_id,
            name=name,
            state_type=state_type,
            screenshot_path=screenshot_path,
            viewport=viewport,
            url=self.page.url,
        )

    async def _find_interactive_selectors(self) -> list[dict[str, Any]]:
        """Find CSS selectors for interactive elements."""
        if not self.page:
            return []

        # Use JavaScript to find interactive elements
        selectors = await self.page.evaluate(
            """
            () => {
                const elements = [];
                const interactiveSelectors = [
                    'button',
                    'a',
                    'input',
                    'select',
                    'textarea',
                    '[role="button"]',
                    '[role="link"]',
                    '[role="menuitem"]',
                    '[role="tab"]',
                    '[onclick]',
                    '[data-testid]',
                ];

                interactiveSelectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach((el, idx) => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            // Generate a unique selector
                            let selector = sel;
                            if (el.id) {
                                selector = `#${el.id}`;
                            } else if (el.dataset.testid) {
                                selector = `[data-testid="${el.dataset.testid}"]`;
                            } else if (el.className && typeof el.className === 'string') {
                                const classes = el.className.split(' ').filter(c => c).slice(0, 2);
                                if (classes.length > 0) {
                                    selector = `${el.tagName.toLowerCase()}.${classes.join('.')}`;
                                }
                            }

                            elements.push({
                                selector: selector,
                                tagName: el.tagName.toLowerCase(),
                                text: el.textContent?.trim().slice(0, 50) || '',
                                bbox: {
                                    x: Math.round(rect.x),
                                    y: Math.round(rect.y),
                                    width: Math.round(rect.width),
                                    height: Math.round(rect.height),
                                },
                                type: el.type || '',
                                ariaLabel: el.getAttribute('aria-label') || '',
                            });
                        }
                    });
                });

                // Deduplicate by selector
                const seen = new Set();
                return elements.filter(e => {
                    if (seen.has(e.selector)) return false;
                    seen.add(e.selector);
                    return true;
                });
            }
        """
        )

        selectors_result: list[dict[str, Any]] = selectors  # type: ignore[assignment]
        return selectors_result

    async def _find_modal_triggers(self) -> list[dict[str, str]]:
        """Find elements that might open modals/dialogs."""
        if not self.page:
            return []

        triggers = await self.page.evaluate(
            """
            () => {
                const triggers = [];
                const patterns = [
                    '[data-modal]',
                    '[aria-haspopup="dialog"]',
                    'button:contains("open")',
                    'button:contains("show")',
                    'button:contains("add")',
                    'button:contains("create")',
                    'button:contains("new")',
                ];

                // Simple text-based matching for buttons
                document.querySelectorAll('button').forEach(btn => {
                    const text = btn.textContent?.toLowerCase() || '';
                    if (['open', 'show', 'add', 'create', 'new', 'edit', 'settings'].some(w => text.includes(w))) {
                        triggers.push({
                            selector: btn.id ? `#${btn.id}` : `button:has-text("${btn.textContent?.trim()}")`,
                            name: text.replace(/[^a-z0-9]/gi, '_').slice(0, 20),
                        });
                    }
                });

                return triggers.slice(0, 5);  // Limit to avoid too many modals
            }
        """
        )

        triggers_result: list[dict[str, str]] = triggers  # type: ignore[assignment]
        return triggers_result

    async def _try_close_modal(self) -> None:
        """Try common patterns to close a modal."""
        if not self.page:
            return

        close_selectors = [
            '[data-dismiss="modal"]',
            'button[aria-label="Close"]',
            'button:has-text("Close")',
            'button:has-text("Cancel")',
            ".modal-close",
            ".close-button",
        ]

        for selector in close_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    await element.click()
                    await self.page.wait_for_timeout(300)
                    return
            except Exception:
                pass

        # Try pressing Escape
        try:
            await self.page.keyboard.press("Escape")
            await self.page.wait_for_timeout(300)
        except Exception:
            pass

    async def _extract_visible_state_images(self, screenshots_dir: Path) -> list[StateImage]:
        """Extract StateImages for all visible interactive elements."""
        images: list[StateImage] = []
        selectors = await self._find_interactive_selectors()

        for selector_info in selectors[:20]:  # Limit to avoid too many
            try:
                imgs = await self.extract_state_images_for_selector(
                    selector_info["selector"],
                    capture_hover=False,
                    capture_focus=False,
                )
                images.extend(imgs)
            except Exception:
                pass

        return images

    async def extract_state_images_for_selector(
        self,
        selector: str,
        capture_hover: bool = True,
        capture_focus: bool = True,
    ) -> list[StateImage]:
        """
        Extract StateImage patterns for a specific CSS selector.

        Captures the element in different visual states and returns
        StateImage objects with precise bounding boxes.
        """
        if not self.page:
            raise RuntimeError("Not connected to page")

        images: list[StateImage] = []

        try:
            element = await self.page.query_selector(selector)
            if not element:
                return images

            # Get element info
            bbox_data = await element.bounding_box()
            if not bbox_data:
                return images

            bbox = BoundingBox(
                x=int(bbox_data["x"]),
                y=int(bbox_data["y"]),
                width=int(bbox_data["width"]),
                height=int(bbox_data["height"]),
            )

            # Get element properties
            props = await self.page.evaluate(
                """
                (selector) => {
                    const el = document.querySelector(selector);
                    if (!el) return null;
                    return {
                        tagName: el.tagName.toLowerCase(),
                        text: el.textContent?.trim().slice(0, 100) || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        role: el.getAttribute('role') || '',
                        type: el.type || '',
                        className: el.className || '',
                    };
                }
                """,
                selector,
            )

            if not props:
                return images

            # Determine element type
            element_type = self._infer_element_type(props)
            image_name = self._generate_image_name(props, selector)
            image_id = f"img_{uuid.uuid4().hex[:8]}"

            # Create StateImage (required fields: id, name, element_type)
            state_image = StateImage(
                id=image_id,
                name=image_name,
                element_type=element_type,
                selector=selector,
                is_interactive=True,
                text_content=props.get("text"),
                aria_label=props.get("ariaLabel"),
                semantic_role=props.get("role"),
                search_region=bbox,
            )

            # Capture normal state pattern
            normal_pattern = await self._capture_element_pattern(element, "normal", bbox)
            if normal_pattern:
                state_image.patterns.append(normal_pattern)

            # Capture hover state
            if capture_hover:
                try:
                    await element.hover()
                    await self.page.wait_for_timeout(100)
                    hover_pattern = await self._capture_element_pattern(element, "hover", bbox)
                    if hover_pattern:
                        state_image.patterns.append(hover_pattern)
                except Exception:
                    pass

            # Capture focus state
            if capture_focus and element_type in ["input", "textarea", "button"]:
                try:
                    await element.focus()
                    await self.page.wait_for_timeout(100)
                    focus_pattern = await self._capture_element_pattern(element, "focus", bbox)
                    if focus_pattern:
                        state_image.patterns.append(focus_pattern)
                except Exception:
                    pass

            # Move mouse away to reset state
            await self.page.mouse.move(0, 0)

            if state_image.patterns:
                images.append(state_image)

        except Exception as e:
            logger.debug(f"Failed to extract StateImage for {selector}: {e}")

        return images

    async def _capture_element_pattern(
        self,
        element: Any,
        pattern_name: str,
        bbox: BoundingBox,
    ) -> ImagePattern | None:
        """Capture a visual pattern for an element."""
        try:
            # Take a screenshot of just this element
            screenshot_bytes = await element.screenshot()

            # Generate hash
            pixel_hash = hashlib.md5(screenshot_bytes, usedforsecurity=False).hexdigest()

            # For now, we don't save individual patterns to disk
            # In a full implementation, you'd save these to the screenshots dir

            return ImagePattern(
                id=f"pattern_{uuid.uuid4().hex[:6]}",
                name=pattern_name,
                pixel_data_path=Path("/dev/null"),  # Placeholder
                pixel_hash=pixel_hash,
                bbox=bbox,
            )

        except Exception as e:
            logger.debug(f"Failed to capture pattern {pattern_name}: {e}")
            return None

    def _infer_element_type(self, props: dict[str, Any]) -> str:
        """Infer the element type from its properties."""
        tag = props.get("tagName", "").lower()
        role = props.get("role", "").lower()
        input_type = props.get("type", "").lower()

        if tag == "button" or role == "button":
            return "button"
        elif tag == "a" or role == "link":
            return "link"
        elif tag == "input":
            if input_type in ["text", "email", "password", "search"]:
                return "input"
            elif input_type == "checkbox":
                return "checkbox"
            elif input_type == "radio":
                return "radio"
            return "input"
        elif tag == "select":
            return "dropdown"
        elif tag == "textarea":
            return "textarea"
        elif role in ["menuitem", "option"]:
            return "menu_item"
        elif role == "tab":
            return "tab"

        return "component"

    def _generate_image_name(self, props: dict[str, Any], selector: str) -> str:
        """Generate a descriptive name for a StateImage."""
        # Try to use text content first
        text = props.get("text", "").strip()
        if text and len(text) < 30:
            # Clean up text for use as name
            name = "".join(c for c in text if c.isalnum() or c == " ")
            name = name.strip().replace(" ", "_")[:20]
            if name:
                return f"{props.get('tagName', 'element')}_{name}"

        # Try aria-label
        aria = props.get("ariaLabel", "").strip()
        if aria:
            name = "".join(c for c in aria if c.isalnum() or c == " ")
            name = name.strip().replace(" ", "_")[:20]
            if name:
                return name

        # Fall back to selector-based name
        if selector.startswith("#"):
            return selector[1:][:30]
        elif "[data-testid=" in selector:
            testid = selector.split('"')[1] if '"' in selector else selector
            return testid[:30]

        return f"element_{uuid.uuid4().hex[:6]}"

    def _enrich_state_image_with_static(
        self,
        image: StateImage,
        static_data: dict[str, Any],
    ) -> StateImage:
        """Enrich a StateImage with information from static analysis."""
        # Try to find matching component from static analysis
        for comp in static_data.get("components", []):
            comp_name_lower = comp["name"].lower()
            image_name_lower = image.name.lower()

            if comp_name_lower in image_name_lower or image_name_lower in comp_name_lower:
                image.component_name = comp["name"]
                image.source_file = Path(comp["file"])
                image.source_line = comp["line"]
                break

        # Try to find matching event handler
        for handler in static_data.get("event_handlers", []):
            if handler["event"] == "click":
                # Could match by selector or component
                pass

        return image

    async def _extract_transitions(
        self,
        config: HybridExtractionConfig,
        static_data: dict[str, Any],
        states: list[State],
    ) -> list[StateTransition]:
        """Extract state transitions from static + runtime data."""
        transitions: list[StateTransition] = []

        # From navigation links in static analysis
        for link in static_data.get("navigation_links", []):
            transition = StateTransition(
                id=f"trans_{uuid.uuid4().hex[:6]}",
                from_state_id=states[0].id if states else "unknown",
                to_state_id="unknown",  # Would need to navigate to determine
                trigger=TransitionTrigger.CLICK,
                navigation_path=link["target"],
                source_file=Path(link["file"]) if link.get("file") else None,
                source_line=link.get("line"),
            )
            transitions.append(transition)

        # From event handlers
        for handler in static_data.get("event_handlers", []):
            if handler["event"] == "click" and handler.get("state_changes"):
                transition = StateTransition(
                    id=f"trans_{uuid.uuid4().hex[:6]}",
                    from_state_id=states[0].id if states else "unknown",
                    to_state_id="unknown",
                    trigger=TransitionTrigger.CLICK,
                    event_handler_name=handler.get("name"),
                    source_file=Path(handler["file"]) if handler.get("file") else None,
                    source_line=handler.get("line"),
                    metadata={"state_changes": handler["state_changes"]},
                )
                transitions.append(transition)

        # Create transitions between parent and child states
        for state in states:
            for child_id in state.child_state_ids:
                transition = StateTransition(
                    id=f"trans_{uuid.uuid4().hex[:6]}",
                    from_state_id=state.id,
                    to_state_id=child_id,
                    trigger=TransitionTrigger.CLICK,
                )
                transitions.append(transition)

        return transitions

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self.page:
            await self.page.close()
            self.page = None

        if self.context:
            await self.context.close()
            self.context = None

        if self.browser:
            await self.browser.close()
            self.browser = None

        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

        if self.dev_process:
            try:
                self.dev_process.terminate()
                self.dev_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dev_process.kill()
                self.dev_process.wait()
            except Exception:
                pass
            self.dev_process = None
