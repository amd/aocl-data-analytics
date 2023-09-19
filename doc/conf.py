# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'AOCL-DA'
copyright = '2023, Advanced Micro Devices, Inc'
author = 'Advanced Micro Devices, Inc'
version = '0.1.0'
release = '0.1.0'


# -- General configuration ---------------------------------------------------
extensions = ['sphinxcontrib.bibtex', 'breathe']
bibtex_bibfiles = ['refs.bib']
bibtex_reference_style = 'author_year'
breathe_default_project = 'aocl-da'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_template']

# -- Options for HTML output -------------------------------------------------

html_theme = 'rocm_docs_theme'
html_theme_options = {
    "link_main_doc": False,
    "flavor": "local",
    "repository_provider" : None,
}
