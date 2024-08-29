# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os, sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.abspath("./_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MemSave PyTorch"
copyright = "2024, Samarth Bhatia, Felix Dangel"
author = "Samarth Bhatia, Felix Dangel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
docsearch_app_id = os.getenv("DOCSEARCH_APP_ID")
docsearch_api_key = os.getenv("DOCSEARCH_API_KEY")
docsearch_index_name = os.getenv("DOCSEARCH_INDEX_NAME")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxemoji.sphinxemoji",
    "sphinx_sitemap",
    "dict2table",
    "sphinx_contributors",
    "sphinx_docsearch",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoclass_content = "both"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
}
# add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = "https://memsave-torch.readthedocs.io/"
html_theme = "shibuya"
html_static_path = ["_static"]
# html_js_files = ['autoscroll.js'] added in the theme itself: https://github.com/lepture/shibuya/issues/30
html_css_files = ["style.css"]

html_theme_options = {
    "accent_color": "teal",
    "github_url": "https://github.com/plutonium-239/memsave_torch",
    "og_image_url": "https://memsave-torch.readthedocs.io/_static/memsave_torch_logo_inv_text_256x256.png",
    "nav_links": [
        {"title": "memsave_torch", "url": "index"},
    ],
    "nav_links_align": "center",
}
html_logo = "_static/memsave_torch_logo_inv.svg"
html_favicon = "_static/favicon.png"

html_context = {
    "source_type": "github",
    "source_user": "plutonium-239",
    "source_repo": "memsave_torch",
    "source_version": "main",  # Optional
    "source_docs_path": "/docs_src/",  # Optional
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "repo-stats-custom.html",
        # "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
        "sidebar_logo.html",
    ]
}
