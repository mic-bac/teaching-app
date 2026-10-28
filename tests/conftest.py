"""Pytest configuration.

Ensures the workspace root (which contains ``Home.py`` and the ``utils`` package)
is importable regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
