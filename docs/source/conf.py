# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'ASSUME'
copyright = '2022-2023 ASSUME Developers'
author = 'ASSUME Developers'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

source_suffix = ".rst"
master_doc = "index"

# -- Options for HTML output ----------------------------------------------
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/assume-framework/assume.git",
    "use_repository_button": True,
    "show_navbar_depth": 2,
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "ASSUME: Agent-Based Electricity Markets Simulation Toolbox"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "ASSUME"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "img/assume_logo.png"

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

# -- Options for EPUB output
epub_show_urls = 'footnote'
