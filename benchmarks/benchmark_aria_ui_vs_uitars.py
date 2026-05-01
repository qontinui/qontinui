"""Benchmark: Aria-UI vs UI-TARS for element grounding.

Compares grounding accuracy and latency between Aria-UI and the existing
UI-TARS integration on representative qontinui target UIs.

Measures:
- Grounding accuracy (distance from expected coordinates)
- Latency (p50, p95, mean)
- VRAM footprint (reported from nvidia-smi)

Prerequisites:
- Aria-UI Docker service running on localhost:8100
  (see docker/aria-ui/docker-compose.yml)
- UI-TARS provider configured (see extraction/runtime/uitars/)
- Screenshot fixtures in benchmarks/screenshots/

Usage:
    python -m benchmarks.benchmark_aria_ui_vs_uitars
    python -m benchmarks.benchmark_aria_ui_vs_uitars --aria-only
    python -m benchmarks.benchmark_aria_ui_vs_uitars --uitars-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark task definitions
# ---------------------------------------------------------------------------

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"


@dataclass
class GroundingTask:
    """A single element-grounding benchmark task."""

    screenshot_path: str
    element_description: str
    expected_x: int
    expected_y: int
    tolerance_px: int = 20
    category: str = "general"


# Representative tasks across common target applications.
# expected_x/y are ground-truth pixel coordinates.
# Add real screenshots to benchmarks/screenshots/ and update these.
BENCHMARK_TASKS: list[GroundingTask] = [
    # --- Notepad ---
    GroundingTask("notepad_file_menu.png", "File menu", 30, 12, 15, "menu"),
    GroundingTask("notepad_save_dialog.png", "Save button", 680, 430, 20, "button"),
    GroundingTask("notepad_edit_area.png", "Text editing area", 400, 300, 50, "region"),
    # --- Chrome ---
    GroundingTask("chrome_address_bar.png", "Address bar", 500, 35, 30, "input"),
    GroundingTask("chrome_new_tab.png", "New tab button", 250, 12, 15, "button"),
    GroundingTask("chrome_settings_menu.png", "Three-dot menu", 1890, 35, 15, "icon"),
    # --- Windows Explorer ---
    GroundingTask("explorer_search.png", "Search box", 1050, 25, 25, "input"),
    GroundingTask("explorer_nav_up.png", "Up arrow navigation", 52, 25, 15, "icon"),
    GroundingTask("explorer_file_item.png", "First file in list", 400, 200, 30, "item"),
    # --- Calculator ---
    GroundingTask("calc_equals.png", "Equals button", 320, 540, 20, "button"),
    GroundingTask("calc_digit_5.png", "Number 5 button", 200, 450, 20, "button"),
    # --- Generic dialog ---
    GroundingTask("dialog_ok.png", "OK button", 350, 280, 20, "button"),
    GroundingTask("dialog_cancel.png", "Cancel button", 450, 280, 20, "button"),
    GroundingTask("dialog_close.png", "Close button (X)", 580, 10, 15, "icon"),
    # --- Taskbar ---
    GroundingTask("taskbar_start.png", "Start button", 24, 1065, 15, "icon"),
    GroundingTask("taskbar_search.png", "Search icon", 60, 1065, 15, "icon"),
    # --- Multi-step ambiguous (context-aware advantage) ---
    GroundingTask("form_submit_1.png", "Submit button", 600, 500, 20, "button"),
    GroundingTask(
        "form_submit_2.png", "Submit button (second form)", 600, 700, 20, "button"
    ),
    GroundingTask(
        "wizard_next_step2.png", "Next button on step 2", 700, 550, 20, "button"
    ),
    GroundingTask(
        "wizard_next_step3.png", "Next button on step 3", 700, 550, 20, "button"
    ),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GroundingAttempt:
    """Result of a single grounding attempt."""

    task: GroundingTask
    backend: str  # "aria_ui" or "uitars"
    predicted_x: int | None = None
    predicted_y: int | None = None
    distance_px: float | None = None
    within_tolerance: bool = False
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results for one backend."""

    backend: str
    total_tasks: int = 0
    successful: int = 0
    within_tolerance: int = 0
    accuracy_pct: float = 0.0
    mean_distance_px: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    attempts: list[GroundingAttempt] = field(default_factory=list)

    def compute(self) -> None:
        """Compute aggregate statistics from individual attempts."""
        self.total_tasks = len(self.attempts)
        successful = [a for a in self.attempts if a.error is None]
        self.successful = len(successful)
        self.within_tolerance = sum(1 for a in successful if a.within_tolerance)
        self.accuracy_pct = (
            (self.within_tolerance / self.total_tasks * 100)
            if self.total_tasks
            else 0.0
        )

        distances = [a.distance_px for a in successful if a.distance_px is not None]
        latencies = [a.latency_ms for a in successful]

        if distances:
            self.mean_distance_px = statistics.mean(distances)
        if latencies:
            self.latency_mean_ms = statistics.mean(latencies)
            self.latency_p50_ms = statistics.median(latencies)
            sorted_lat = sorted(latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            self.latency_p95_ms = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]


# ---------------------------------------------------------------------------
# Grounding backends
# ---------------------------------------------------------------------------


async def ground_with_aria_ui(
    screenshot_bytes: bytes,
    task: GroundingTask,
    endpoint: str = "http://localhost:8100",
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> GroundingAttempt:
    """Run a single grounding task against Aria-UI."""
    from qontinui.healing.aria_ui_client import AriaUIClient
    from qontinui.healing.healing_types import HealingContext

    client = AriaUIClient(endpoint=endpoint, timeout=120.0)
    context = HealingContext(
        original_description=task.element_description,
        screenshot_shape=(screen_height, screen_width),
    )

    start = time.perf_counter()
    try:
        location = client.find_element(screenshot_bytes, context)
    except Exception as e:
        return GroundingAttempt(
            task=task,
            backend="aria_ui",
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if location is None:
        return GroundingAttempt(
            task=task,
            backend="aria_ui",
            latency_ms=elapsed_ms,
            error="not_found",
        )

    dist = math.sqrt(
        (location.x - task.expected_x) ** 2 + (location.y - task.expected_y) ** 2
    )
    return GroundingAttempt(
        task=task,
        backend="aria_ui",
        predicted_x=location.x,
        predicted_y=location.y,
        distance_px=dist,
        within_tolerance=dist <= task.tolerance_px,
        latency_ms=elapsed_ms,
    )


async def ground_with_uitars(
    screenshot_bytes: bytes,
    task: GroundingTask,
) -> GroundingAttempt:
    """Run a single grounding task against UI-TARS.

    Uses the existing UITARSExecutor infrastructure.
    """
    try:
        import numpy as np

        from qontinui.extraction.runtime.uitars.config import UITARSSettings
        from qontinui.extraction.runtime.uitars.executor import UITARSExecutor
        from qontinui.extraction.runtime.uitars.provider import create_provider

        settings = UITARSSettings()
        provider = create_provider(settings)
        executor = UITARSExecutor(provider=provider, settings=settings)

        # Decode screenshot to numpy for UITARSExecutor
        import cv2

        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        screenshot = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        start = time.perf_counter()
        result = await executor.ground_element(
            element_name=task.element_description,
            screenshot=screenshot,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not result or not result.found:
            return GroundingAttempt(
                task=task,
                backend="uitars",
                latency_ms=elapsed_ms,
                error="not_found",
            )

        dist = math.sqrt(
            (result.x - task.expected_x) ** 2 + (result.y - task.expected_y) ** 2
        )
        return GroundingAttempt(
            task=task,
            backend="uitars",
            predicted_x=result.x,
            predicted_y=result.y,
            distance_px=dist,
            within_tolerance=dist <= task.tolerance_px,
            latency_ms=elapsed_ms,
        )

    except Exception as e:
        return GroundingAttempt(
            task=task,
            backend="uitars",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    run_aria: bool = True,
    run_uitars: bool = True,
    aria_endpoint: str = "http://localhost:8100",
) -> dict[str, BenchmarkResults]:
    """Run the full benchmark suite.

    Args:
        run_aria: Whether to benchmark Aria-UI.
        run_uitars: Whether to benchmark UI-TARS.
        aria_endpoint: Aria-UI vLLM server endpoint.

    Returns:
        Dict mapping backend name to BenchmarkResults.
    """
    results: dict[str, BenchmarkResults] = {}

    # Filter tasks to only those with available screenshots
    available_tasks = []
    for task in BENCHMARK_TASKS:
        screenshot_path = SCREENSHOTS_DIR / task.screenshot_path
        if screenshot_path.exists():
            available_tasks.append((task, screenshot_path.read_bytes()))
        else:
            logger.warning(f"Screenshot not found: {screenshot_path}")

    if not available_tasks:
        logger.error(
            f"No screenshots found in {SCREENSHOTS_DIR}. "
            "Add PNG screenshots and update BENCHMARK_TASKS with correct coordinates."
        )
        return results

    print(f"\nRunning benchmark with {len(available_tasks)} tasks\n")

    if run_aria:
        aria_results = BenchmarkResults(backend="aria_ui")
        print("=== Aria-UI ===")
        for task, screenshot_bytes in available_tasks:
            attempt = await ground_with_aria_ui(
                screenshot_bytes, task, endpoint=aria_endpoint
            )
            aria_results.attempts.append(attempt)
            status = (
                "HIT"
                if attempt.within_tolerance
                else ("MISS" if attempt.error is None else "ERR")
            )
            dist_str = (
                f"{attempt.distance_px:.1f}px"
                if attempt.distance_px is not None
                else "N/A"
            )
            print(
                f"  [{status}] {task.element_description:<35} dist={dist_str:<10} {attempt.latency_ms:.0f}ms"
            )
        aria_results.compute()
        results["aria_ui"] = aria_results

    if run_uitars:
        uitars_results = BenchmarkResults(backend="uitars")
        print("=== UI-TARS ===")
        for task, screenshot_bytes in available_tasks:
            attempt = await ground_with_uitars(screenshot_bytes, task)
            uitars_results.attempts.append(attempt)
            status = (
                "HIT"
                if attempt.within_tolerance
                else ("MISS" if attempt.error is None else "ERR")
            )
            dist_str = (
                f"{attempt.distance_px:.1f}px"
                if attempt.distance_px is not None
                else "N/A"
            )
            print(
                f"  [{status}] {task.element_description:<35} dist={dist_str:<10} {attempt.latency_ms:.0f}ms"
            )
        uitars_results.compute()
        results["uitars"] = uitars_results

    return results


def print_comparison(results: dict[str, BenchmarkResults]) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 70)

    headers = ["Metric"] + list(results.keys())
    rows: list[list[str]] = [
        ["Tasks run"] + [str(r.total_tasks) for r in results.values()],
        ["Successful"] + [str(r.successful) for r in results.values()],
        ["Within tolerance"] + [str(r.within_tolerance) for r in results.values()],
        ["Accuracy %"] + [f"{r.accuracy_pct:.1f}%" for r in results.values()],
        ["Mean distance (px)"]
        + [f"{r.mean_distance_px:.1f}" for r in results.values()],
        ["Latency mean (ms)"] + [f"{r.latency_mean_ms:.0f}" for r in results.values()],
        ["Latency p50 (ms)"] + [f"{r.latency_p50_ms:.0f}" for r in results.values()],
        ["Latency p95 (ms)"] + [f"{r.latency_p95_ms:.0f}" for r in results.values()],
    ]

    col_widths = [
        max(len(h), max(len(r) for r in col))
        for h, col in zip(headers, zip(*rows, strict=True), strict=True)
    ]
    col_widths = [max(20, w) for w in col_widths]

    header_line = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths, strict=True)
    )
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_widths, strict=True)))

    print()


def save_results(results: dict[str, BenchmarkResults], output_path: Path) -> None:
    """Save results to JSON for later analysis."""
    data: dict[str, Any] = {}
    for name, result in results.items():
        r = {
            "backend": result.backend,
            "total_tasks": result.total_tasks,
            "successful": result.successful,
            "within_tolerance": result.within_tolerance,
            "accuracy_pct": result.accuracy_pct,
            "mean_distance_px": result.mean_distance_px,
            "latency_mean_ms": result.latency_mean_ms,
            "latency_p50_ms": result.latency_p50_ms,
            "latency_p95_ms": result.latency_p95_ms,
            "attempts": [
                {
                    "element": a.task.element_description,
                    "screenshot": a.task.screenshot_path,
                    "expected": [a.task.expected_x, a.task.expected_y],
                    "predicted": (
                        [a.predicted_x, a.predicted_y]
                        if a.predicted_x is not None
                        else None
                    ),
                    "distance_px": a.distance_px,
                    "within_tolerance": a.within_tolerance,
                    "latency_ms": a.latency_ms,
                    "error": a.error,
                }
                for a in result.attempts
            ],
        }
        data[name] = r

    output_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Aria-UI vs UI-TARS")
    parser.add_argument(
        "--aria-only", action="store_true", help="Only benchmark Aria-UI"
    )
    parser.add_argument(
        "--uitars-only", action="store_true", help="Only benchmark UI-TARS"
    )
    parser.add_argument(
        "--aria-endpoint",
        default="http://localhost:8100",
        help="Aria-UI vLLM server endpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results_aria_ui_vs_uitars.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_aria = not args.uitars_only
    run_uitars = not args.aria_only

    results = asyncio.run(
        run_benchmark(
            run_aria=run_aria,
            run_uitars=run_uitars,
            aria_endpoint=args.aria_endpoint,
        )
    )

    if results:
        print_comparison(results)
        save_results(results, args.output)
    else:
        print("No results. Ensure screenshots are in benchmarks/screenshots/.")


if __name__ == "__main__":
    main()
