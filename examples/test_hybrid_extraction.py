#!/usr/bin/env python3
"""
Test script for hybrid (white_box) extraction on qontinui-web frontend.

This script tests the hybrid translation functionality which combines:
1. Static source code analysis (TypeScript/React/Next.js)
2. Runtime Playwright extraction (on compiled/running app)
3. Correlation of static and runtime results

Usage:
    python test_hybrid_extraction.py

Requirements:
    - qontinui-web frontend running on localhost:3001
    - qontinui library installed
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_hybrid_extraction():
    """Test hybrid extraction on qontinui-web frontend."""
    from qontinui.extraction import (
        ExtractionConfig,
        ExtractionMode,
        ExtractionOrchestrator,
        ExtractionTarget,
        FrameworkType,
        StateStructure,
    )

    # Configuration for qontinui-web frontend
    frontend_path = Path(__file__).parent.parent.parent / "qontinui-web" / "frontend"
    frontend_url = "http://localhost:3001"

    logger.info(f"Frontend path: {frontend_path}")
    logger.info(f"Frontend URL: {frontend_url}")

    # Verify frontend path exists
    if not frontend_path.exists():
        logger.error(f"Frontend path does not exist: {frontend_path}")
        return None

    # Check if package.json exists to confirm it's a valid project
    package_json = frontend_path / "package.json"
    if not package_json.exists():
        logger.error(f"package.json not found at {package_json}")
        return None

    logger.info("Creating extraction configuration...")

    # Create extraction target
    target = ExtractionTarget(
        project_path=frontend_path,
        url=frontend_url,
        framework=FrameworkType.NEXT,  # Next.js framework
    )

    # Create extraction config for WHITE_BOX mode
    config = ExtractionConfig(
        target=target,
        mode=ExtractionMode.WHITE_BOX,  # Hybrid: static + runtime
        viewports=[(1920, 1080)],
        capture_hover_states=False,  # Simpler for testing
        capture_focus_states=False,
        capture_scroll_states=False,
        max_interaction_depth=1,  # Just the landing page
        correlation_threshold=0.7,
        require_correlation=False,  # Don't fail on low correlation
        timeout_seconds=120,
    )

    logger.info(f"Extraction mode: {config.mode.value}")
    logger.info("Creating orchestrator...")

    # Create orchestrator
    orchestrator = ExtractionOrchestrator()

    logger.info("Starting extraction...")
    logger.info("-" * 60)

    try:
        # Run extraction
        result = await orchestrator.extract(config)

        logger.info("-" * 60)
        logger.info("EXTRACTION RESULTS")
        logger.info("-" * 60)

        logger.info(f"Extraction ID: {result.extraction_id}")
        logger.info(f"Framework: {result.framework.value}")
        logger.info(f"Mode: {result.mode.value}")

        # Static analysis results
        if result.static_analysis:
            sa = result.static_analysis
            logger.info("\nStatic Analysis:")
            logger.info(f"  - Files analyzed: {sa.analyzed_files}")
            logger.info(f"  - Components found: {len(sa.components)}")
            logger.info(f"  - Routes found: {len(sa.routes)}")
            logger.info(f"  - State definitions: {len(sa.state_definitions)}")
            logger.info(f"  - Event handlers: {len(sa.event_handlers)}")
            logger.info(f"  - Navigation flows: {len(sa.navigation_flows)}")
            logger.info(f"  - Duration: {sa.analysis_duration_ms:.2f}ms")

            if sa.components:
                logger.info("\n  Components:")
                for comp in sa.components[:5]:  # First 5
                    logger.info(f"    - {comp.get('name', 'Unknown')}")

            if sa.routes:
                logger.info("\n  Routes:")
                for route in sa.routes[:5]:  # First 5
                    logger.info(f"    - {route.get('path', 'Unknown')}")

        # Runtime extraction results
        if result.runtime_extraction:
            re = result.runtime_extraction
            logger.info("\nRuntime Extraction:")
            logger.info(f"  - Pages visited: {re.pages_visited}")
            logger.info(f"  - States found: {len(re.states)}")
            logger.info(f"  - Elements found: {len(re.elements)}")
            logger.info(f"  - Transitions found: {len(re.transitions)}")
            logger.info(f"  - Screenshots: {len(re.screenshots)}")
            logger.info(f"  - Duration: {re.extraction_duration_ms:.2f}ms")

        # Correlated states
        logger.info(f"\nCorrelated States: {len(result.states)}")
        for state in result.states[:5]:  # First 5
            logger.info(f"  - {state.name} (confidence: {state.confidence:.2f})")
            if hasattr(state, "correlation_score") and state.correlation_score:
                logger.info(f"    correlation: {state.correlation_score:.2f}")
            if hasattr(state, "source_file") and state.source_file:
                logger.info(f"    source: {state.source_file}")

        # Transitions
        logger.info(f"\nTransitions: {len(result.transitions)}")
        for trans in result.transitions[:5]:  # First 5
            from_id = getattr(trans, "from_state_id", "?")
            to_id = getattr(trans, "to_state_id", "?")
            logger.info(f"  - {from_id} -> {to_id}")

        # Errors and warnings
        if result.errors:
            logger.warning(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                logger.warning(f"  - {error}")

        if result.warnings:
            logger.info(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings[:5]:
                logger.info(f"  - {warning}")

        # Create StateStructure from result
        logger.info("\n" + "-" * 60)
        logger.info("CREATING STATE STRUCTURE")
        logger.info("-" * 60)

        state_structure = StateStructure.from_extraction_result(
            result,
            source_id="qontinui-web-frontend",
        )

        logger.info(f"StateStructure ID: {state_structure.id}")
        logger.info(f"States: {len(state_structure.states)}")
        logger.info(f"Transitions: {len(state_structure.transitions)}")
        logger.info(f"Elements: {len(state_structure.elements)}")
        logger.info(f"Screenshots: {len(state_structure.screenshots)}")
        logger.info(f"Sources: {state_structure.get_sources()}")

        # Check for disjoint trees
        components = state_structure.get_connected_components()
        logger.info(f"Connected components (disjoint trees): {len(components)}")

        # Serialize to dict
        structure_dict = state_structure.to_dict()
        logger.info(f"\nSerialized structure keys: {list(structure_dict.keys())}")

        return state_structure

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return None


async def test_static_only():
    """Test static-only extraction (no runtime)."""
    from qontinui.extraction import (
        ExtractionConfig,
        ExtractionMode,
        ExtractionOrchestrator,
        ExtractionTarget,
        FrameworkType,
    )

    frontend_path = Path(__file__).parent.parent.parent / "qontinui-web" / "frontend"

    logger.info("\n" + "=" * 60)
    logger.info("STATIC ONLY EXTRACTION TEST")
    logger.info("=" * 60)

    target = ExtractionTarget(
        project_path=frontend_path,
        framework=FrameworkType.NEXT,
    )

    config = ExtractionConfig(
        target=target,
        mode=ExtractionMode.STATIC_ONLY,
        timeout_seconds=60,
    )

    orchestrator = ExtractionOrchestrator()

    try:
        result = await orchestrator.extract(config)

        logger.info(f"States from static analysis: {len(result.states)}")
        logger.info(f"Transitions from static analysis: {len(result.transitions)}")

        if result.static_analysis:
            logger.info(f"Components: {len(result.static_analysis.components)}")
            logger.info(f"Routes: {len(result.static_analysis.routes)}")

        return result

    except Exception as e:
        logger.error(f"Static extraction failed: {e}", exc_info=True)
        return None


def main():
    """Run the tests."""
    print("=" * 60)
    print("HYBRID EXTRACTION TEST")
    print("=" * 60)
    print()
    print("This test requires:")
    print("  1. qontinui-web frontend running on localhost:3001")
    print("  2. qontinui library with extraction module")
    print()
    print("To start the frontend:")
    print("  cd qontinui-web/frontend && npm run dev")
    print()
    print("-" * 60)

    # Run static-only first (doesn't need running server)
    asyncio.run(test_static_only())

    print()
    print("-" * 60)
    print("Now testing full hybrid extraction (needs frontend running)...")
    print("-" * 60)

    # Run full hybrid extraction
    result = asyncio.run(test_hybrid_extraction())

    if result:
        print()
        print("=" * 60)
        print("TEST COMPLETE - State structure created successfully")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("TEST FAILED - Check logs above for errors")
        print("=" * 60)


if __name__ == "__main__":
    main()
