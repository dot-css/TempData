# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'TempData'
copyright = '2024, TempData Team'
author = 'TempData Team'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "tasklist",
]

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Master document
master_doc = 'index'

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = [
    'pandas', 
    'numpy', 
    'faker', 
    'pytz', 
    'click', 
    'tqdm', 
    'openpyxl'
]