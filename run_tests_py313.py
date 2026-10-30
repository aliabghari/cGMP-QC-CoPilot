#!/usr/bin/env python3
"""
Python 3.13 Pytest Runner (`run_tests_py313.py`).
Executes the full test suite under Python 3.13 with autograd/PyTorch active.
"""

import sys
from pathlib import Path

# Append site-packages fallback if pure-python pytest is located in 3.14
import os
custom_site = os.environ.get("CUSTOM_SITE_PACKAGES")
if custom_site:
    site_path = Path(custom_site)
    if site_path.exists() and str(site_path) not in sys.path:
        sys.path.append(str(site_path))

import pluggy
pluggy.PluginManager.load_setuptools_entrypoints = lambda self, name: 0

import pytest

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/", "-v"]
    sys.exit(pytest.main(args))
