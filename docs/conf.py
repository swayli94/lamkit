from __future__ import annotations

import os
import re
import sys
import warnings
from datetime import datetime
from importlib.util import find_spec

# Anchor paths to this file so builds work from any cwd (e.g. repo root:
# ``sphinx-build -b html docs docs/_build/html``).
_CONF_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_CONF_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

_INIT_PY = os.path.join(SRC_ROOT, "lamkit", "__init__.py")
with open(_INIT_PY, encoding="utf-8") as _f:
    _init_src = _f.read()
_match = re.search(
    r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
    _init_src,
    re.MULTILINE,
)
release = _match.group(1) if _match else "0.0.0"
version = release.rpartition(".")[0] if release.count(".") >= 2 else release

project = "lamkit"
author = "Runze Li"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"
# Furo is optional at parse time: editors may analyze conf.py without docs extras installed.
if find_spec("furo") is not None:
    html_theme = "furo"
else:
    html_theme = "alabaster"
    warnings.warn(
        "Sphinx theme 'furo' is not installed; using bundled 'alabaster'. "
        'Install docs extras: pip install -e ".[docs]" or pip install furo',
        UserWarning,
        stacklevel=1,
    )
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Autodoc imports lamkit modules, which require NumPy, pandas, Matplotlib, SciPy, etc.
# Install with: pip install -e ".[docs]"  (see docs/installation.rst).
