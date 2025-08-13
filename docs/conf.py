#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration file for Sphinx documentation"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
]

__date__: str = "2025-08-13"
__status__: str = "Development"
__version__: str = "0.0.1"
__license__: str = "GPL-3.0"

# Minimal Sphinx configuration with Napoleon (NumPy/Google) docstring support
import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(".."))

project = "Configurable Spectrograms"
author = "Ev Hansen"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Napoleon settings for NumPy-style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# Mock heavy imports so docs build without full environment
autodoc_mock_imports = [
    "cdflib",
    "numpy",
    "matplotlib",
    "tqdm",
    "pandas",
]

html_theme = "alabaster"

# Keep warnings for missing references noisy to help catch issues during build
nitpicky = False
