"""Shared pytest fixtures for RASVC-X tests.

Ensures both ``src/`` (for rasvcx package) and the repo root (for the
evaluation/ package) are importable without requiring an editable install,
so these tests can run directly against the src-layout package on any
platform including Windows.
"""

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def _hermetic_rasvcx_env(monkeypatch: pytest.MonkeyPatch):
    """Run every test with no RASVCX_* variables inherited from the shell.

    The application reads its mode, API key and paths from the environment.
    A developer's shell (or a loaded .env) must not decide whether a test
    passes, and a test must never pick up a real API key by accident.
    Tests that need a variable set it themselves.
    """
    for name in [k for k in os.environ if k.startswith("RASVCX_")]:
        monkeypatch.delenv(name, raising=False)
    yield
