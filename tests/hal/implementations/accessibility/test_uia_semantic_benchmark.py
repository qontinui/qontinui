"""Benchmark tests for the progressive ratio sieve in fuzzy_match_nodes().

The three-pass sieve (real_quick_ratio → quick_ratio → ratio) short-circuits
expensive full-ratio computation when the cheap upper bounds already fall below
the cutoff.  These benchmarks verify the speedup exists and is measurable at
the ratio-computation level.

Design note
-----------
The sieve benefit lives *inside* the (keyword, node_name) pair loop — it is the
saving of `ratio()` calls when `real_quick_ratio()` or `quick_ratio()` already
rules out a match.  The rest of `fuzzy_match_nodes` (flatten, spatial-label
inference, multi-mode regex transforms, dict lookups) is overhead shared by both
paths and is not what we are benchmarking here.

The apples-to-apples comparison is therefore:

    naive:  SequenceMatcher(None, a, b).ratio()          — always pays full cost
    sieve:  _sieve_ratio(a, b, cutoff)                   — bails after cheap passes

We run both over the same set of (keyword, node_name) pairs extracted from the
300-node snapshot to make the comparison fair.
"""

import sys
import time
from pathlib import Path

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from difflib import SequenceMatcher

from qontinui_schemas.accessibility import (
    AccessibilityBackend,
    AccessibilityBounds,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySnapshot,
)

from qontinui.hal.implementations.accessibility.uia_semantic import (
    SemanticSearchCache,
    _extract_name_keywords,
    _flatten_nodes,
    _sieve_ratio,
    fuzzy_match_nodes,
)

# ---------------------------------------------------------------------------
# Helpers shared with test_uia_semantic.py
# ---------------------------------------------------------------------------


def _make_node(
    ref: str = "@e1",
    role: AccessibilityRole = AccessibilityRole.BUTTON,
    name: str | None = None,
    value: str | None = None,
    automation_id: str | None = None,
    is_interactive: bool = True,
    children: list[AccessibilityNode] | None = None,
    bounds: AccessibilityBounds | None = None,
) -> AccessibilityNode:
    """Helper to build an AccessibilityNode for testing."""
    return AccessibilityNode(
        ref=ref,
        role=role,
        name=name,
        value=value,
        automation_id=automation_id,
        is_interactive=is_interactive,
        children=children or [],
        bounds=bounds,
    )


def _make_snapshot(nodes: list[AccessibilityNode]) -> AccessibilitySnapshot:
    """Wrap a list of nodes under a root and return a snapshot."""
    root = _make_node(
        ref="@root",
        role=AccessibilityRole.APPLICATION,
        name="Test App",
        is_interactive=False,
        children=nodes,
    )
    return AccessibilitySnapshot(
        root=root,
        timestamp=time.time(),
        backend=AccessibilityBackend.UIA,
        total_nodes=1 + len(nodes),
        interactive_nodes=sum(1 for n in nodes if n.is_interactive),
    )


# ---------------------------------------------------------------------------
# Snapshot fixture
# ---------------------------------------------------------------------------


def _make_large_snapshot(num_nodes: int = 300) -> AccessibilitySnapshot:
    """Create a snapshot with many varied nodes to stress the sieve."""
    common_names = [
        "Save",
        "Cancel",
        "OK",
        "Submit",
        "Close",
        "Open",
        "New",
        "Delete",
        "Edit",
        "Copy",
        "Paste",
        "Cut",
        "Undo",
        "Redo",
        "Help",
        "About",
        "Settings",
        "Preferences",
        "File",
        "View",
    ]
    long_names = [
        "Save Document As PDF",
        "Open Recent File",
        "Export To Spreadsheet",
        "Import Configuration File",
        "Print Preview Document",
        "Close Without Saving",
        "Find And Replace Text",
        "Format Selected Paragraph",
        "Insert Table From File",
        "Navigate To Home Screen",
        "Zoom In To Fit Window",
        "Show All Hidden Items",
        "Collapse All Tree Nodes",
        "Expand Selected Subtree",
        "Reset To Default Values",
    ]
    roles_cycle = [
        AccessibilityRole.BUTTON,
        AccessibilityRole.TEXTBOX,
        AccessibilityRole.LINK,
        AccessibilityRole.MENUITEM,
        AccessibilityRole.CHECKBOX,
    ]

    nodes: list[AccessibilityNode] = []
    for i in range(num_nodes):
        style = i % 4
        if style == 0:
            name = common_names[i % len(common_names)]
        elif style == 1:
            noun = ["Button", "Item", "Entry", "Option", "Field"][i % 5]
            name = f"{noun} {i // 4 + 1}"
        elif style == 2:
            name = long_names[i % len(long_names)]
        else:
            base = common_names[i % len(common_names)]
            name = f"{base} {i}"

        role = roles_cycle[i % len(roles_cycle)]
        ref = f"@e{i + 1}"
        nodes.append(_make_node(ref=ref, role=role, name=name))

    return _make_snapshot(nodes)


# ---------------------------------------------------------------------------
# Benchmark test class
# ---------------------------------------------------------------------------


class TestFuzzyMatchPerformance:
    """Performance benchmarks for the progressive ratio sieve in fuzzy_match_nodes."""

    def test_progressive_sieve_speedup(self):
        """Progressive ratio sieve should be >= 2.5x faster than naive ratio on 300 nodes.

        Measures speedup at the ratio-computation level — the direct benefit of
        the three-pass sieve (real_quick_ratio → quick_ratio → ratio).  With
        min_score=0.6, approximately 94%+ of pairs are pruned by the cheap
        passes, making the full ratio() call rare.

        The naive baseline runs the identical loop over the same (keyword,
        node_name) pairs but always calls SequenceMatcher.ratio() directly
        without any short-circuit.
        """
        snapshot = _make_large_snapshot(300)
        interactive_nodes = _flatten_nodes(snapshot.root, interactive_only=True)
        node_names = [(n.name or "").lower() for n in interactive_nodes if n.name]

        queries = [
            "Save Document button",
            "the search text field",
            "Cancel dialog",
            "preferences settings menu",
            "copy clipboard",
        ]
        # min_score=0.6 means the sieve cutoff is 0.6 — at this threshold,
        # real_quick_ratio + quick_ratio prune ~94%+ of pairs on this tree.
        min_score = 0.6
        iterations = 500

        kws_list = [_extract_name_keywords(q) for q in queries]

        # Warm up
        for kws, q in zip(kws_list, queries, strict=False):
            q_lower = q.lower()
            for name in node_names:
                for kw in kws:
                    _sieve_ratio(kw, name, min_score)
                _sieve_ratio(q_lower, name, min_score)

        # ----------------------------------------------------------------
        # Benchmark: three-pass sieve
        # ----------------------------------------------------------------
        start = time.perf_counter()
        for _ in range(iterations):
            for kws, q in zip(kws_list, queries, strict=False):
                q_lower = q.lower()
                for name in node_names:
                    for kw in kws:
                        _sieve_ratio(kw, name, min_score)
                    _sieve_ratio(q_lower, name, min_score)
        sieve_time = time.perf_counter() - start

        # ----------------------------------------------------------------
        # Benchmark: naive — same pairs, always calls full ratio()
        # ----------------------------------------------------------------
        start = time.perf_counter()
        for _ in range(iterations):
            for kws, q in zip(kws_list, queries, strict=False):
                q_lower = q.lower()
                for name in node_names:
                    for kw in kws:
                        SequenceMatcher(None, kw, name).ratio()
                    SequenceMatcher(None, q_lower, name).ratio()
        naive_time = time.perf_counter() - start

        speedup = naive_time / sieve_time
        print(
            f"\nRatio-level benchmark (min_score={min_score}, {iterations} iters × "
            f"{len(queries)} queries × {len(node_names)} nodes):\n"
            f"  naive={naive_time:.3f}s  sieve={sieve_time:.3f}s  speedup={speedup:.1f}x"
        )

        assert speedup >= 2.5, (
            f"Expected >= 2.5x speedup from progressive ratio sieve, got {speedup:.1f}x. "
            f"naive={naive_time:.3f}s sieve={sieve_time:.3f}s"
        )

    def test_ratio_cache_speedup(self):
        """Using ratio cache should speed up repeated queries with overlapping node names."""
        snapshot = _make_large_snapshot(300)
        cache = SemanticSearchCache()

        queries = ["Save button", "Save Document", "Save Changes", "Save File", "Save As"]

        # First pass: cold cache
        start = time.perf_counter()
        for q in queries:
            fuzzy_match_nodes(q, snapshot, min_score=0.3, cache=cache)
        cold_time = time.perf_counter() - start

        # Second pass: warm cache (ratio entries populated from first pass)
        start = time.perf_counter()
        for q in queries:
            fuzzy_match_nodes(q, snapshot, min_score=0.3, cache=cache)
        warm_time = time.perf_counter() - start

        print(
            f"\nCache benchmark: cold={cold_time:.3f}s  warm={warm_time:.3f}s  "
            f"ratio_cache_size={len(cache._ratio_cache)}"
        )

        assert warm_time < cold_time, (
            f"Warm cache should be faster than cold: "
            f"cold={cold_time:.3f}s warm={warm_time:.3f}s"
        )

    def test_sieve_prunes_high_fraction_of_pairs(self):
        """With min_score=0.6 on 300 varied nodes, the sieve prunes >= 90% of pairs.

        This validates that the speedup is structural — most node names are
        dissimilar enough to the query keywords that real_quick_ratio() or
        quick_ratio() eliminates them before the expensive full ratio() runs.
        """
        snapshot = _make_large_snapshot(300)
        interactive_nodes = _flatten_nodes(snapshot.root, interactive_only=True)
        node_names = [(n.name or "").lower() for n in interactive_nodes if n.name]

        queries = [
            "Save Document button",
            "the search text field",
            "Cancel dialog",
            "preferences settings menu",
            "copy clipboard",
        ]
        min_score = 0.6
        kws_list = [_extract_name_keywords(q) for q in queries]

        pruned = 0
        total = 0
        for kws, q in zip(kws_list, queries, strict=False):
            q_lower = q.lower()
            for name in node_names:
                for kw in kws:
                    total += 1
                    sm = SequenceMatcher(None, kw, name)
                    if sm.real_quick_ratio() < min_score or sm.quick_ratio() < min_score:
                        pruned += 1
                total += 1
                sm = SequenceMatcher(None, q_lower, name)
                if sm.real_quick_ratio() < min_score or sm.quick_ratio() < min_score:
                    pruned += 1

        prune_pct = 100.0 * pruned / total
        print(
            f"\nSieve prune rate (min_score={min_score}): " f"{pruned}/{total} = {prune_pct:.1f}%"
        )

        assert prune_pct >= 90.0, (
            f"Expected sieve to prune >= 90% of pairs at min_score={min_score}, "
            f"got {prune_pct:.1f}% ({pruned}/{total})"
        )

    def test_sieve_no_match_per_call_latency(self):
        """On a 300-node tree with no strong matches, each fuzzy_match_nodes call
        completes in under 100 ms.
        """
        snapshot = _make_large_snapshot(300)

        # A query with no matches in our node tree
        nonsense_query = "xyzzy quantum frobnitz"

        start = time.perf_counter()
        for _ in range(50):
            fuzzy_match_nodes(nonsense_query, snapshot, min_score=0.5)
        elapsed = time.perf_counter() - start
        per_call_ms = (elapsed / 50) * 1000

        print(f"\nNo-match latency: {per_call_ms:.1f} ms per call (50 calls, 300 nodes)")

        assert per_call_ms < 100, (
            f"Expected < 100 ms per call on 300-node no-match query, " f"got {per_call_ms:.1f} ms"
        )
