"""Shared pytest fixtures for RASVC-X tests.

Ensures ``src/`` is importable without requiring an editable install, so
these tests can run directly against the src-layout package.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))