# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MemSave PyTorch'
copyright = '2024, Samarth Bhatia, Felix Dangel'
author = 'Samarth Bhatia, Felix Dangel'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_copybutton', 'sphinx.ext.autosummary', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'

intersphinx_mapping = {
	'python': ('https://docs.python.org/3', None),
	'torch': ('https://pytorch.org/docs/stable/', None),

}
# add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
html_js_files = ['autoscroll.js']
html_css_files = ['style.css']

html_theme_options = {
  "accent_color": "pink",
}