"""Pytest configuration: headless matplotlib for CI and local runs."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
