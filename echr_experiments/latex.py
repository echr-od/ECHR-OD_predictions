import jinja2
import os
from jinja2 import Template


def initialize_latex_env():
	latex_jinja_env = jinja2.Environment(
		block_start_string = '\BLOCK{',
		block_end_string = '}',
		variable_start_string = '\VAR{',
		variable_end_string = '}',
		comment_start_string = '\#{',
		comment_end_string = '}',
		line_statement_prefix = '%%',
		line_comment_prefix = '%#',
		trim_blocks = True,
		autoescape = False,
		loader = jinja2.FileSystemLoader(os.path.abspath('.'))
	)
	return latex_jinja_env
