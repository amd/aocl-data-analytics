# Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#


# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
import sys
import os
from docutils import nodes
project = 'AOCL-DA'
copyright = '2026, Advanced Micro Devices, Inc'
author = 'Advanced Micro Devices, Inc'
version = ''
release = '5.2.2'

# -- Get doc working for Python ----------------------------------------------
# Add to PYTHONPATH
sys.path.insert(0, os.path.relpath('../python_interface/'))
autodoc_mock_imports = ['aoclda._aoclda', 'numpy']

# Add a custom 'nref' role that renders plain text with no cross-reference ---
def nref_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Role that renders text as-is with no cross-referencing."""
    if options is None:
        options = {}
    if content is None:
        content = []
    node = nodes.inline(rawtext, text, classes=['nref'])
    return [node], []

def setup(app):
    app.add_role('nref', nref_role)
    return {
        'version': release,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

# -- General configuration ---------------------------------------------------
extensions = ['sphinxcontrib.bibtex', 'breathe', 'sphinx.ext.napoleon',
              'sphinx.ext.autodoc', 'sphinx_design']
bibtex_bibfiles = ['refs.bib']
bibtex_reference_style = 'author_year'

# -- Option for generating output conditional on cmake INTERNAL_DOC variable -
exclude_patterns = ['**/doc/trees_forests/df_intro.rst']

# -- Add any paths that contain templates here, relative to this directory. --
templates_path = ['_template']

# -- MathJax config ----------------------------------------------------------
mathjax3_config = {
    'chtml': {
        'mtextInheritFont': 'true',
    }
}


# -- Options for HTML output -------------------------------------------------

html_theme = 'rocm_docs_theme'
html_theme_options = {
    "link_main_doc": False,
    "flavor": "local",
    "repository_provider": None,
    "navigation_with_keys": False,
    "navigation_depth": 1,
}

latex_elements = {
    "preamble": '''
\\setcounter{tocdepth}{2}
'''
}
