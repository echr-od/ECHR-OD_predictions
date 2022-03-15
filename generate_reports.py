#!/usr/bin/python
import argparse
import json
import os
from os import listdir, path
import copy
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

from echr.utils.folders import make_build_folder
from echr.utils.logger import getlogger
from echr.utils.cli import TAB
from rich.markdown import Markdown
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.tree import Tree
from echr_experiments.config import BINARY_DESC_OUTPUT_FILE, MULTICLASS_DESC_OUTPUT_FILE, MULTILABEL_DESC_OUTPUT_FILE

from math import floor, log10

def run(console, build, force):
    __console = console
    global print
    print = __console.print


    N = 2
    print(Markdown("- **Prepare dataset descriptions**"))
    with open(BINARY_DESC_OUTPUT_FILE, 'r') as f:
        binary_desc = json.load(f)

    binary_desc = pd.read_json(BINARY_DESC_OUTPUT_FILE)
    binary_desc = binary_desc.reindex(sorted(binary_desc.columns, key=lambda x: int(x) if x != 'p1-1' else 999), axis=1)
    binary_desc.columns = [f'Article {c}' for c in binary_desc.columns]
    binary_desc = binary_desc.T
    prev = binary_desc['prevalence']
    binary_desc = binary_desc.astype(int)
    
    binary_desc['prevalence'] = prev.apply(lambda x: round(x, N - int(floor(log10(abs(x))))))
    binary_desc.columns = [c.replace('_', ' ').title() for c in binary_desc.columns]
    binary_desc.to_latex(Path(ANALYSIS_PATH) / 'tables' / 'binary_datasets_summary.tex', 
        bold_rows=True, label='table:binary_datasets', caption='Datasets description for binary classification.')



    print(Markdown("- **Prepare dataset descriptions**"))
    with open(MULTICLASS_DESC_OUTPUT_FILE, 'r') as f:
        multiclass_desc = json.load(f)
    multiclass_desc = pd.DataFrame(multiclass_desc['Multiclass'].values())

    multiclass_desc['sort'] = multiclass_desc['Article'].apply(lambda x: int(x) if x != 'p1-1' else 999)
    multiclass_desc['Article'] = multiclass_desc['Article'].apply(lambda x: f'Article {x}')
    multiclass_desc = multiclass_desc.sort_values(by="sort")
    multiclass_desc['Prev. Violation'] = multiclass_desc['Violation'] / multiclass_desc['Size'].sum()
    multiclass_desc['Prev. No-Violation'] = multiclass_desc['No-Violation'] / multiclass_desc['Size'].sum()

    multiclass_desc['Prevalence'] = multiclass_desc['Prevalence'].apply(lambda x: round(x, N - int(floor(log10(abs(x))))))
    multiclass_desc['Prev. Violation'] = multiclass_desc['Prev. Violation'].apply(lambda x: round(x, N - int(floor(log10(abs(x))))))
    multiclass_desc['Prev. No-Violation'] = multiclass_desc['Prev. No-Violation'].apply(lambda x: round(x, N - 1 - int(floor(log10(abs(x))))))

    multiclass_desc['Violation'] = multiclass_desc.apply(lambda x: "{} ({:.3f})".format(x['Violation'], x['Prev. Violation']), axis=1)
    multiclass_desc['No-Violation'] = multiclass_desc.apply(lambda x: "{} ({:.3f})".format(x['No-Violation'], x['Prev. No-Violation']), axis=1)
    multiclass_desc = multiclass_desc[['Article', 'Size', 'Violation', 'No-Violation', 'Prevalence']]
    multiclass_desc = multiclass_desc.rename(columns={'Article': ""})
    multiclass_desc.to_latex(Path(ANALYSIS_PATH) / 'tables' / 'multiclass_datasets_summary.tex', 
        bold_rows=True, index=False, label='table:multiclass_datasets', 
        caption='Datasets description for multiclass classification.')



    print(Markdown("- **Prepare dataset descriptions**"))
    with open(MULTILABEL_DESC_OUTPUT_FILE, 'r') as f:
        multilabel_desc = json.load(f)
    print(multilabel_desc)
    multilabel_desc = pd.DataFrame(multilabel_desc['Multilabel'].values())
    multilabel_desc = multilabel_desc[multilabel_desc['Size'] > 100 ]
    multilabel_desc['sort'] = multilabel_desc['Article'].apply(lambda x: int(x) if not x.startswith('p') else 999)
    multilabel_desc['Article'] = multilabel_desc['Article'].apply(lambda x: f'Article {x}')
    multilabel_desc = multilabel_desc.sort_values(by="sort")
    print(multilabel_desc)
    multilabel_desc['Prev. Violation'] = multilabel_desc['Violation'] / multilabel_desc['Size'].sum()
    multilabel_desc['Prev. No-Violation'] = multilabel_desc['No-Violation'] / multilabel_desc['Size'].sum()

    multilabel_desc['Prevalence'] = multilabel_desc['Prevalence'].apply(lambda x: round(x, N - int(floor(log10(abs(x))))) if x > 0 else 0)
    multilabel_desc['Prev. Violation'] = multilabel_desc['Prev. Violation'].apply(lambda x: round(x, N - int(floor(log10(abs(x))))) if x > 0 else 0)
    multilabel_desc['Prev. No-Violation'] = multilabel_desc['Prev. No-Violation'].apply(lambda x: round(x, N - 1 - int(floor(log10(abs(x))))) if x > 0 else 0)

    multilabel_desc['Violation'] = multilabel_desc.apply(lambda x: "{} ({:.3f})".format(x['Violation'], x['Prev. Violation']), axis=1)
    multilabel_desc['No-Violation'] = multilabel_desc.apply(lambda x: "{} ({:.3f})".format(x['No-Violation'], x['Prev. No-Violation']), axis=1)
    multilabel_desc = multilabel_desc[['Article', 'Size', 'Violation', 'No-Violation', 'Prevalence']]
    multilabel_desc = multilabel_desc.rename(columns={'Article': ""})
    multilabel_desc.to_latex(Path(ANALYSIS_PATH) / 'tables' / 'multilabel_datasets_summary.tex', 
        bold_rows=True, index=False, label='table:multilabel_datasets', 
        caption='Datasets description for multilabel classification.')

    


def main(args):
    console = Console(record=True)
    run(console, args.build, args.force)


def parse_args(parser):
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate post-experiments reports')
    parser.add_argument('--build', type=str, default="./build/echr_database/")
    parser.add_argument('-f', '--force', action='store_true')
    args = parse_args(parser)

    main(args)
