"""Pytest configuration for coordinates tests.

NOTE: Due to the cv2 DLL import issue on Windows, pytest cannot properly
collect these tests because the main qontinui.__init__.py imports cv2
through the discovery module chain.

To run the coordinate tests, use the standalone test runner:

    python tests/coordinates/test_runner_standalone.py

The standalone runner works by inserting the src path before any imports,
allowing the coordinates module to be tested in isolation.
"""

# Pytest configuration kept minimal - use standalone runner instead
