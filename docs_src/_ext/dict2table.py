from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.parsers import RSTParser
from docutils.utils import new_document
from docutils.frontend import OptionParser

from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from importlib import import_module
from typing import Dict, List, Tuple

def format_function(x):
    if x.__name__ == '<lambda>':
        return x
    module = x.__module__
    if module.startswith('torchvision'):
        module = '.'.join(x.__module__.split('.')[:-1])
    content =  module + "." + x.__name__ 
    return f':func:`{content}`'

class Dict2Table(SphinxDirective):
    has_content = True
    required__arguments = 1
    option_spec = {
        "filter": directives.unchanged_required,
        "filter-out": directives.unchanged_required,
        "caption": directives.unchanged_required,
    }

    def run(self):
        module_path, member_name = self.content[0].rsplit(".", 1)
        member_data = getattr(import_module(module_path), member_name)
        assert isinstance(member_data, dict)
        if "filter" in self.options:
            member_data = {
                k: v for k, v in member_data.items() if self.options.get("filter") in k
            }
        if "filter-out" in self.options:
            member_data = {
                k: v
                for k, v in member_data.items()
                if self.options.get("filter-out") not in k
            }

        table_head = (
            f'.. csv-table:: {self.options.get("caption")}\n'
            + '  :header: "Model Name", "Model Function"\n'
            + "  :widths: 4, 6\n\n"
        )

        table_str = "\n".join([f'  "{k}", "{format_function(v)}"' for k, v in member_data.items()])
        table_str = table_head + table_str
        columns = {"model_name": 4, "function": 6}
        # table, tablegroup = self._PrepareTable(member_data, )

        # import pdb; pdb.set_trace()
        main_node = nodes.paragraph()
        self.state.nested_parse(StringList(table_str.split('\n')), 0, main_node)

        # return self.parse_rst(table_str)

        return [main_node]

    def parse_rst(self, text):
        parser = RSTParser()
        parser.set_application(self.env.app)

        settings = OptionParser(
            defaults=self.env.settings,
            components=(RSTParser,),
            read_config_files=True,
        ).get_default_values()
        document = new_document("<rst-doc>", settings=settings)
        parser.parse(text, document)
        return document.children

    # def _PrepareTable(self, columns: Dict[str, int], identifier: str, classes: List[str]) -> Tuple[nodes.table, nodes.tgroup]:
    #     table = nodes.table("", identifier=identifier, classes=classes)

    #     tableGroup = nodes.tgroup(cols=(len(columns)))
    #     table += tableGroup

    #     tableRow = nodes.row()
    #     for columnTitle, width in columns.items():
    #         tableGroup += nodes.colspec(colwidth=width)
    #         tableRow += nodes.entry("", nodes.paragraph(text=columnTitle))

    #     tableGroup += nodes.thead("", tableRow)

    #     return table, tableGroup


def setup(app: Sphinx):
    app.add_directive("dict2table", Dict2Table)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
