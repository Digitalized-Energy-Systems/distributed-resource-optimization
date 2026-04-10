# Configuration file for the Sphinx documentation builder.

project = "distributed-resource-optimization"
copyright = "2024, Rico Schrage"
author = "Rico Schrage"
version = release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "dark_logo": "logo.svg",
    "light_logo": "logo.svg",
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/Digitalized-Energy-Systems/mango-optimization/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "top_of_page_buttons": ["view", "edit"],
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
