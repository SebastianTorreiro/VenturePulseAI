"""Pytest bootstrap. Markers and paths live in pyproject.toml.

The sys.path insert below is a fallback for environments running an old
pytest without `pythonpath` ini support; pytest >= 7 does not need it.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
