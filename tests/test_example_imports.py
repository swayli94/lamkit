"""
Examples are valid scripts: import paths and main helpers run without error.

Does not assert on figure pixel values; uses Agg backend from conftest.
"""

import os
import runpy
import sys


def _run_script(path: str) -> None:
    path = os.path.abspath(path)
    old_path = sys.path[:]
    src = os.path.abspath(os.path.join(os.path.dirname(path), "..", "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        sys.path[:] = old_path


def test_example_7_lekhnitskii_runs():
    root = os.path.dirname(os.path.dirname(__file__))
    _run_script(os.path.join(root, "example", "7-lekhnitskii", "example_unloaded_hole.py"))


def test_example_8_laminate_open_hole_runs():
    root = os.path.dirname(os.path.dirname(__file__))
    _run_script(os.path.join(root, "example", "8-laminate-open-hole", "example_laminate_open_hole.py"))
