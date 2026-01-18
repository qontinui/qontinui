"""
DOM stability detection and dynamic content handling.

This module provides utilities to wait for DOM stability before extraction,
detect mutations, and handle lazy-loaded content. Essential for reliable
extraction from dynamic, JavaScript-heavy web applications.

Key Features
------------
- **Stability Detection**: Wait until DOM stops mutating
- **Mutation Tracking**: Record what changed via MutationObserver
- **Lazy Loading**: Scroll to trigger content loading
- **Load More Buttons**: Auto-click pagination buttons
- **Change Detection**: Compare page state over time

Classes
-------
DOMStabilityWaiter
    Wait for DOM to stabilize before extraction.
LazyContentLoader
    Handle lazy-loaded content via scroll/click.
ContentChangeDetector
    Detect significant content changes.
DOMSnapshot
    Point-in-time DOM state for comparison.
StabilityResult
    Result of stability waiting.
MutationRecord
    Individual DOM mutation details.

Functions
---------
wait_for_stable_extraction
    Convenience function for stability waiting.
load_lazy_content
    Convenience function for lazy loading.

Usage Examples
--------------
Wait for DOM stability::

    from qontinui.extraction.web import wait_for_stable_extraction

    result = await wait_for_stable_extraction(
        page,
        stability_ms=500,  # No mutations for 500ms = stable
        max_wait_ms=5000,  # Give up after 5 seconds
    )
    if result.stable:
        print(f"Stable after {result.wait_time_ms}ms")
        print(f"Mutations observed: {result.mutation_count}")
    else:
        print("Page did not stabilize")

With DOMStabilityWaiter::

    from qontinui.extraction.web import DOMStabilityWaiter

    waiter = DOMStabilityWaiter(
        stability_threshold_ms=100,  # Fast stability check
        max_wait_ms=3000,
    )
    result = await waiter.wait_for_stability(page)

Load lazy content::

    from qontinui.extraction.web import load_lazy_content

    # Scroll to load infinite scroll content
    stats = await load_lazy_content(page, scroll=True)
    print(f"Loaded {stats['scroll']['elements_loaded']} new elements")

    # Click "Load More" buttons
    stats = await load_lazy_content(page, click_load_more=True)

With LazyContentLoader::

    from qontinui.extraction.web import LazyContentLoader

    loader = LazyContentLoader(
        scroll_pause_ms=500,
        max_scrolls=10,
    )
    stats = await loader.load_all_content(page, scroll_direction="down")
    stats = await loader.click_load_more(page, max_clicks=5)

Detect content changes::

    from qontinui.extraction.web import ContentChangeDetector

    detector = ContentChangeDetector(change_threshold=0.1)  # 10% change
    await detector.set_baseline(page)

    # ... do something ...

    changed, details = await detector.has_significant_change(page)
    if changed:
        print(f"Content changed: {details['element_change_ratio']:.1%}")

See Also
--------
- interactive_element_extractor: Uses stability in extract_with_stability()
- hybrid_extractor: Uses stability before context extraction
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class DOMSnapshot:
    """Snapshot of DOM state for comparison."""

    timestamp: float
    element_count: int
    content_hash: str
    scroll_position: tuple[int, int]
    document_height: int

    def differs_from(self, other: "DOMSnapshot") -> bool:
        """Check if this snapshot differs significantly from another."""
        if self.element_count != other.element_count:
            return True
        if self.content_hash != other.content_hash:
            return True
        return False


@dataclass
class MutationRecord:
    """Record of a DOM mutation."""

    timestamp: float
    mutation_type: str  # "childList", "attributes", "characterData"
    target_tag: str
    target_id: str | None
    added_nodes: int
    removed_nodes: int


@dataclass
class StabilityResult:
    """Result of waiting for DOM stability."""

    stable: bool
    wait_time_ms: float
    mutation_count: int
    final_snapshot: DOMSnapshot | None
    mutations: list[MutationRecord] = field(default_factory=list)


class DOMStabilityWaiter:
    """
    Wait for DOM to stabilize before extraction.

    Uses MutationObserver (via CDP) to detect when DOM changes
    have settled, ensuring extractions capture complete content.

    Performance optimizations:
    - Default stability threshold reduced to 100ms (from 500ms)
    - Early exit if page is already stable (no pending network/animations)
    - Configurable early exit check
    """

    def __init__(
        self,
        stability_threshold_ms: int = 100,
        max_wait_ms: int = 5000,
        poll_interval_ms: int = 50,
        check_early_exit: bool = True,
    ):
        """
        Initialize the stability waiter.

        Args:
            stability_threshold_ms: Time without mutations to consider stable (default: 100ms)
            max_wait_ms: Maximum time to wait before giving up (default: 5000ms)
            poll_interval_ms: How often to check for stability (default: 50ms)
            check_early_exit: Whether to check for early exit conditions (default: True)
        """
        self.stability_threshold_ms = stability_threshold_ms
        self.max_wait_ms = max_wait_ms
        self.poll_interval_ms = poll_interval_ms
        self.check_early_exit = check_early_exit

    async def _check_page_ready(self, page: Page) -> bool:
        """
        Check if page is already stable (no pending activity).

        This enables early exit without waiting for the full stability threshold.
        Checks:
        - document.readyState is 'complete'
        - No pending animations
        - No active network requests (via Performance API)

        Returns:
            True if page appears stable and ready
        """
        try:
            is_ready = await page.evaluate(
                """
                () => {
                    // Check document ready state
                    if (document.readyState !== 'complete') return false;

                    // Check for running animations (if Web Animations API available)
                    if (typeof document.getAnimations === 'function') {
                        const animations = document.getAnimations();
                        const runningAnimations = animations.filter(
                            a => a.playState === 'running' || a.playState === 'pending'
                        );
                        if (runningAnimations.length > 0) return false;
                    }

                    // Check for pending resource loads via Performance API
                    if (window.performance && window.performance.getEntriesByType) {
                        const resources = window.performance.getEntriesByType('resource');
                        const recentResources = resources.filter(r => {
                            // Resources loaded in the last 100ms might still be processing
                            return (performance.now() - r.responseEnd) < 100;
                        });
                        if (recentResources.length > 0) return false;
                    }

                    return true;
                }
                """
            )
            return is_ready
        except Exception:
            return False

    async def wait_for_stability(self, page: Page) -> StabilityResult:
        """
        Wait for DOM to stabilize.

        Returns when no DOM mutations occur for stability_threshold_ms,
        or when max_wait_ms is exceeded.

        Includes early exit optimization: if the page appears ready and stable
        on first check, returns immediately without waiting.

        Args:
            page: Playwright Page to monitor

        Returns:
            StabilityResult with stability status and metrics
        """
        start_time = time.time()

        # Early exit check - if page is already stable, return immediately
        if self.check_early_exit:
            is_ready = await self._check_page_ready(page)
            if is_ready:
                # Take a quick snapshot and return
                snapshot = await self._take_snapshot(page)
                wait_time = (time.time() - start_time) * 1000
                logger.debug(f"Early exit: page already stable after {wait_time:.0f}ms")
                return StabilityResult(
                    stable=True,
                    wait_time_ms=wait_time,
                    mutation_count=0,
                    final_snapshot=snapshot,
                    mutations=[],
                )

        mutations: list[MutationRecord] = []
        last_mutation_time = start_time

        # Set up mutation observer via page.evaluate
        observer_script = """
        () => {
            window.__domMutations = [];
            window.__mutationObserver = new MutationObserver((mutations) => {
                for (const mutation of mutations) {
                    window.__domMutations.push({
                        type: mutation.type,
                        targetTag: mutation.target.tagName || 'unknown',
                        targetId: mutation.target.id || null,
                        addedNodes: mutation.addedNodes.length,
                        removedNodes: mutation.removedNodes.length,
                        timestamp: Date.now()
                    });
                }
            });
            window.__mutationObserver.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: true,
                characterData: true
            });
        }
        """

        try:
            await page.evaluate(observer_script)
        except Exception as e:
            logger.warning(f"Failed to set up mutation observer: {e}")
            # Fall back to snapshot-based detection
            return await self._wait_with_snapshots(page)

        try:
            while True:
                elapsed_ms = (time.time() - start_time) * 1000

                if elapsed_ms >= self.max_wait_ms:
                    logger.debug(f"Max wait time exceeded ({self.max_wait_ms}ms)")
                    break

                # Check for new mutations
                new_mutations = await page.evaluate(
                    """
                    () => {
                        const mutations = window.__domMutations || [];
                        window.__domMutations = [];
                        return mutations;
                    }
                """
                )

                if new_mutations:
                    last_mutation_time = time.time()
                    for m in new_mutations:
                        mutations.append(
                            MutationRecord(
                                timestamp=m["timestamp"] / 1000,
                                mutation_type=m["type"],
                                target_tag=m["targetTag"],
                                target_id=m["targetId"],
                                added_nodes=m["addedNodes"],
                                removed_nodes=m["removedNodes"],
                            )
                        )

                # Check if stable
                time_since_mutation_ms = (time.time() - last_mutation_time) * 1000
                if time_since_mutation_ms >= self.stability_threshold_ms:
                    logger.debug(
                        f"DOM stable after {elapsed_ms:.0f}ms "
                        f"({len(mutations)} mutations)"
                    )
                    break

                await asyncio.sleep(self.poll_interval_ms / 1000)

        finally:
            # Clean up observer
            try:
                await page.evaluate(
                    """
                    () => {
                        if (window.__mutationObserver) {
                            window.__mutationObserver.disconnect();
                            delete window.__mutationObserver;
                            delete window.__domMutations;
                        }
                    }
                """
                )
            except Exception:
                pass

        # Take final snapshot
        final_snapshot = await self._take_snapshot(page)
        wait_time = (time.time() - start_time) * 1000

        return StabilityResult(
            stable=wait_time < self.max_wait_ms,
            wait_time_ms=wait_time,
            mutation_count=len(mutations),
            final_snapshot=final_snapshot,
            mutations=mutations,
        )

    async def _wait_with_snapshots(self, page: Page) -> StabilityResult:
        """Fallback: wait for stability using snapshot comparison."""
        start_time = time.time()
        last_snapshot = await self._take_snapshot(page)
        last_change_time = start_time
        mutation_count = 0

        while True:
            elapsed_ms = (time.time() - start_time) * 1000

            if elapsed_ms >= self.max_wait_ms:
                break

            await asyncio.sleep(self.poll_interval_ms / 1000)

            current_snapshot = await self._take_snapshot(page)

            if current_snapshot.differs_from(last_snapshot):
                last_change_time = time.time()
                mutation_count += 1
                last_snapshot = current_snapshot

            time_since_change_ms = (time.time() - last_change_time) * 1000
            if time_since_change_ms >= self.stability_threshold_ms:
                break

        wait_time = (time.time() - start_time) * 1000

        return StabilityResult(
            stable=wait_time < self.max_wait_ms,
            wait_time_ms=wait_time,
            mutation_count=mutation_count,
            final_snapshot=last_snapshot,
            mutations=[],
        )

    async def _take_snapshot(self, page: Page) -> DOMSnapshot:
        """Take a DOM snapshot for comparison."""
        try:
            info = await page.evaluate(
                """
                () => {
                    const elements = document.querySelectorAll('*');
                    const texts = [];
                    for (const el of elements) {
                        if (el.textContent) {
                            texts.push(el.tagName + ':' + el.textContent.substring(0, 50));
                        }
                    }
                    return {
                        elementCount: elements.length,
                        contentSample: texts.slice(0, 100).join('|'),
                        scrollX: window.scrollX || 0,
                        scrollY: window.scrollY || 0,
                        documentHeight: document.documentElement.scrollHeight || 0
                    };
                }
            """
            )

            content_hash = hashlib.md5(info["contentSample"].encode()).hexdigest()[:16]

            return DOMSnapshot(
                timestamp=time.time(),
                element_count=info["elementCount"],
                content_hash=content_hash,
                scroll_position=(info["scrollX"], info["scrollY"]),
                document_height=info["documentHeight"],
            )

        except Exception as e:
            logger.warning(f"Failed to take DOM snapshot: {e}")
            return DOMSnapshot(
                timestamp=time.time(),
                element_count=0,
                content_hash="",
                scroll_position=(0, 0),
                document_height=0,
            )


class LazyContentLoader:
    """
    Handle lazy-loaded content by triggering load events.

    Supports:
    - Scroll-triggered lazy loading
    - Intersection observer triggers
    - Explicit load more buttons
    """

    def __init__(
        self,
        scroll_pause_ms: int = 500,
        max_scrolls: int = 10,
    ):
        """
        Initialize the lazy content loader.

        Args:
            scroll_pause_ms: Time to wait after each scroll
            max_scrolls: Maximum number of scroll operations
        """
        self.scroll_pause_ms = scroll_pause_ms
        self.max_scrolls = max_scrolls
        self.stability_waiter = DOMStabilityWaiter(
            stability_threshold_ms=300,
            max_wait_ms=3000,
        )

    async def load_all_content(
        self,
        page: Page,
        scroll_direction: str = "down",
    ) -> dict[str, Any]:
        """
        Scroll through page to trigger lazy loading.

        Args:
            page: Playwright Page
            scroll_direction: "down" or "up"

        Returns:
            Dict with loading statistics
        """
        initial_snapshot = await self.stability_waiter._take_snapshot(page)
        scroll_count = 0
        elements_loaded = 0

        for i in range(self.max_scrolls):
            # Scroll
            if scroll_direction == "down":
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
            else:
                await page.evaluate("window.scrollBy(0, -window.innerHeight)")

            # Wait for content to load
            await asyncio.sleep(self.scroll_pause_ms / 1000)
            result = await self.stability_waiter.wait_for_stability(page)

            scroll_count += 1

            # Check if we've reached the end
            current_snapshot = result.final_snapshot
            if current_snapshot:
                new_elements = (
                    current_snapshot.element_count - initial_snapshot.element_count
                )
                elements_loaded = max(elements_loaded, new_elements)

                # Check if at bottom/top
                at_boundary = await page.evaluate(
                    f"""
                    () => {{
                        const atBottom = (window.innerHeight + window.scrollY) >=
                                        document.documentElement.scrollHeight - 10;
                        const atTop = window.scrollY <= 10;
                        return {'{"direction": "down"}' if scroll_direction == "down" else '{"direction": "up"}'}.direction === "down" ? atBottom : atTop;
                    }}
                """
                )

                if at_boundary:
                    logger.debug(f"Reached page boundary after {scroll_count} scrolls")
                    break

        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")

        return {
            "scroll_count": scroll_count,
            "elements_loaded": elements_loaded,
            "initial_element_count": initial_snapshot.element_count,
            "final_element_count": (
                current_snapshot.element_count if current_snapshot else 0
            ),
        }

    async def click_load_more(
        self,
        page: Page,
        button_selectors: list[str] | None = None,
        max_clicks: int = 5,
    ) -> dict[str, Any]:
        """
        Click "Load More" buttons to load additional content.

        Args:
            page: Playwright Page
            button_selectors: CSS selectors for load more buttons
            max_clicks: Maximum number of clicks

        Returns:
            Dict with loading statistics
        """
        if button_selectors is None:
            button_selectors = [
                "button:has-text('Load More')",
                "button:has-text('Show More')",
                "button:has-text('View More')",
                "a:has-text('Load More')",
                "[class*='load-more']",
                "[class*='show-more']",
            ]

        initial_snapshot = await self.stability_waiter._take_snapshot(page)
        click_count = 0

        for _ in range(max_clicks):
            clicked = False

            for selector in button_selectors:
                try:
                    button = page.locator(selector).first
                    if await button.is_visible():
                        await button.click()
                        clicked = True
                        click_count += 1

                        # Wait for content to load
                        await self.stability_waiter.wait_for_stability(page)
                        break

                except Exception:
                    continue

            if not clicked:
                break

        final_snapshot = await self.stability_waiter._take_snapshot(page)

        return {
            "click_count": click_count,
            "initial_element_count": initial_snapshot.element_count,
            "final_element_count": final_snapshot.element_count,
            "elements_added": final_snapshot.element_count
            - initial_snapshot.element_count,
        }


class ContentChangeDetector:
    """
    Detect significant content changes between extractions.

    Useful for:
    - Detecting when page needs re-extraction
    - Identifying dynamic vs static content regions
    - Tracking content freshness
    """

    def __init__(
        self,
        change_threshold: float = 0.1,  # 10% change
    ):
        """
        Initialize the change detector.

        Args:
            change_threshold: Fraction of content that must change to trigger
        """
        self.change_threshold = change_threshold
        self.baseline_snapshot: DOMSnapshot | None = None

    async def set_baseline(self, page: Page) -> DOMSnapshot:
        """Capture baseline snapshot for comparison."""
        waiter = DOMStabilityWaiter()
        self.baseline_snapshot = await waiter._take_snapshot(page)
        return self.baseline_snapshot

    async def has_significant_change(self, page: Page) -> tuple[bool, dict[str, Any]]:
        """
        Check if page has changed significantly from baseline.

        Args:
            page: Playwright Page

        Returns:
            Tuple of (has_changed, change_details)
        """
        if not self.baseline_snapshot:
            await self.set_baseline(page)
            return False, {"reason": "baseline_set"}

        waiter = DOMStabilityWaiter()
        current = await waiter._take_snapshot(page)

        # Calculate change metrics
        element_diff = abs(
            current.element_count - self.baseline_snapshot.element_count
        )
        element_change_ratio = element_diff / max(
            self.baseline_snapshot.element_count, 1
        )

        hash_changed = current.content_hash != self.baseline_snapshot.content_hash

        details = {
            "element_count_before": self.baseline_snapshot.element_count,
            "element_count_after": current.element_count,
            "element_change_ratio": element_change_ratio,
            "content_hash_changed": hash_changed,
        }

        has_changed = (
            element_change_ratio >= self.change_threshold or hash_changed
        )

        return has_changed, details


async def wait_for_stable_extraction(
    page: Page,
    stability_ms: int = 100,
    max_wait_ms: int = 3000,
    check_early_exit: bool = True,
) -> StabilityResult:
    """
    Convenience function to wait for DOM stability.

    Performance optimized defaults:
    - stability_ms: 100ms (reduced from 500ms)
    - max_wait_ms: 3000ms (reduced from 5000ms)
    - check_early_exit: True (enables immediate return if page is ready)

    Args:
        page: Playwright Page
        stability_ms: Time without mutations for stability (default: 100ms)
        max_wait_ms: Maximum wait time (default: 3000ms)
        check_early_exit: Whether to check for early exit (default: True)

    Returns:
        StabilityResult with stability status
    """
    waiter = DOMStabilityWaiter(
        stability_threshold_ms=stability_ms,
        max_wait_ms=max_wait_ms,
        check_early_exit=check_early_exit,
    )
    return await waiter.wait_for_stability(page)


async def load_lazy_content(
    page: Page,
    scroll: bool = True,
    click_load_more: bool = False,
) -> dict[str, Any]:
    """
    Convenience function to load lazy content.

    Args:
        page: Playwright Page
        scroll: Whether to scroll to trigger lazy loading
        click_load_more: Whether to click "Load More" buttons

    Returns:
        Dict with loading statistics
    """
    loader = LazyContentLoader()
    results: dict[str, Any] = {}

    if scroll:
        results["scroll"] = await loader.load_all_content(page)

    if click_load_more:
        results["load_more"] = await loader.click_load_more(page)

    return results
