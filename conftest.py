"""
conftest.py
================================================================
WHY THIS FILE EXISTS:
pytest's behavior around Python's import path (sys.path) differs
depending on how it's invoked:

  - `python -m pytest`  -> the current directory is automatically
    added to sys.path (because that's how `-m` works).
  - `pytest` (the bare console command) -> the current directory
    is NOT automatically added.

Our tests do `from api.app import app`, which requires the project
root (the folder containing both `api/` and `tests/`) to be on
sys.path. Without this file, running the bare `pytest` command
fails with `ModuleNotFoundError: No module named 'api'`, even
though the exact same command works if you type
`python -m pytest` instead.

Placing an (even empty) conftest.py at the project root guarantees
pytest adds this folder to sys.path during test collection,
regardless of which of the two invocation styles is used. This
makes the test suite work the same way for every student, no
matter how they type the command.
================================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
