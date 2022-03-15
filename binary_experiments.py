#!/usr/bin/python
import argparse
from collections import Counter
import json
import os
from os import listdir, path
import copy
from pathlib import Path
import re
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

from sklearn.preprocessing import MultiLabelBinarizer



import numpy as np
import pandas as pd
import json
from sklearn.model_selection import cross_validate, TimeSeriesSplit, StratifiedKFold

from echr_experiments.config import ROUND_DIGITS, \
                                    SEED, \
                                    BINARY_OUTPUT_FILE, \
                                    BINARY_ARTICLES, \
                                    BINARY_FLAVORS, \
                                    K_FOLD, \
                                    BINARY_CLASSIFIERS, \
                                    DEFAULT_FEATURE_THRESHOLD, \
                                    AS_TIME_SERIES, \
                                    BINARY_DESC_OUTPUT_FILE
from echr_experiments.format import format_filter_output, format_method_output
from echr_experiments.data import load_ECHR_instance, generate_datasets_descriptors
from echr_experiments.scorers import make_scorers, process_score
from echr_experiments.utils import update_classifier_result, \
                                   update_dataset_filter_result, \
                                   update_dataset_result, \
                                   update_dataset_metadata, \
                                   update_article_desc

seed = SEED #random.randint(0,10000)
result_file = BINARY_OUTPUT_FILE
articles = BINARY_ARTICLES
flavors = BINARY_FLAVORS
as_time_series = AS_TIME_SERIES
k_fold = K_FOLD
feature_threshold = DEFAULT_FEATURE_THRESHOLD
classifiers = BINARY_CLASSIFIERS



import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


log = getlogger()


def generate_outcomes_data(y_file, outcome_to_id, filter_threshold=100):
    # Generate hot-one outcome matrix
    with open(y_file) as file:
        f = lambda x: {x.split(':')[0]:x.split(':')[1]}
        outcomes = file.readlines()
        outcomes = pd.DataFrame(outcomes)
        outcomes[0] = outcomes[0].apply(lambda x: x.strip().split())
        outcomes['caseid'] = outcomes[0].apply(lambda x: x[0])
        outcomes[0] = outcomes[0].apply(lambda x: x[1:])

        # Generate hot-one outcome matrix
        mlb = MultiLabelBinarizer(sparse_output=True)
        outcomes = outcomes.join(
                    pd.DataFrame.sparse.from_spmatrix(
                        mlb.fit_transform(outcomes.pop(0)),
                        index=outcomes.index,
                        columns=mlb.classes_))

    # Remove articles with not enough labels
    to_drop = []
    for c in outcomes.columns:
        if c != 'caseid':
            art, val = c.split(':')
            if f'{art}:0' not in outcomes.columns:
                to_drop.append(f'{art}:1')
            if f'{art}:1' not in outcomes.columns:
                to_drop.append(f'{art}:0')
            
            if f'{art}:0' in outcomes.columns and f'{art}:1'  in outcomes.columns:
                sum_0 = outcomes[f'{art}:0'].sum()
                sum_1 = outcomes[f'{art}:1'].sum()
                if sum_0 + sum_1 < filter_threshold:
                    to_drop.append(c)

    outcomes.drop(columns=to_drop, inplace=True)

    return outcomes


def map_outcome(art, x):
            if x[f'{art}:1'] == 1:
                return 1 
            elif x[f'{art}:0'] == 1:
                return 0
            else: 
                return -1


def load_dataset(X_file, min_threshold=0):
    with open(X_file) as file:
        f = lambda x: {x.split(':')[0]:x.split(':')[1]}
        X = file.readlines()
        X = pd.DataFrame(X)
        X[0] = X[0].apply(lambda x: sorted(x.strip().split()))

        # Generate hot-one outcome matrix
        mlb = MultiLabelBinarizer(sparse_output=True)
        X = X.join(
                    pd.DataFrame.sparse.from_spmatrix(
                        mlb.fit_transform(X.pop(0)),
                        index=X.index,
                        columns=mlb.classes_))
        return X


def run(console, build, title, doc_ids=None, force=False):
    __console = console
    global print
    print = __console.print

    outcomes_path = 'data/input/datasets/'
    raw_outcome_file = Path(outcomes_path) / 'outcomes.txt'
    outcome_matrix_file = Path(outcomes_path) / 'outcomes_matrix.csv'

    print(Markdown("- **Prepare outcome matrix**"))
    OUTCOME_TO_ID = 'data/input/datasets/outcomes_variables.json'
    with open(OUTCOME_TO_ID, 'r') as f:
        outcome_to_id = json.load(f)

    
    if not os.path.isfile(outcome_matrix_file) or force:
        print(TAB + '> Generate the outcome matrix [green][DONE]')
        outcomes_matrix = generate_outcomes_data(raw_outcome_file, outcome_to_id, filter_threshold=100)
        print(outcomes_matrix)
        outcomes_matrix.to_csv(outcome_matrix_file)
    else:
        print(TAB + '> Load the outcome matrix [green][DONE]')
    outcome_matrix = pd.read_csv(outcome_matrix_file)

    print(Markdown("- **Experiment summary**"))
    FLAVORS = {'Descriptive only': 'descriptive.txt', 'Bag-of-Words only': 'BoW.txt', 'Descriptive and Bag-of-Words': 'descriptive+BoW.txt'}
    articles = {art:art_id for art, art_id in outcome_to_id.items() if f'{art_id}:1' in outcome_matrix.columns}
    print(f"  | Flavors: {len(FLAVORS)}")
    print(f"  | Articles: {len(articles)}")
    print(f"  | Methods: {len(classifiers)}")
    print(f"  = {len(FLAVORS) * len(articles) * len(classifiers)} cross-validation procedures")
    print(f"  = Take some :coffee: or :tea: and  relax")

    try:
        f = open (result_file, "r")
        exp_results = json.loads(f.read())
        print(TAB + '> Load existing results [green][DONE]')
    except Exception as e:
        exp_results = {}
        print(TAB + '> No previous results [green][DONE]')


    table = Table(title="Cross-Validation Summary")

    table.add_column("Flavor", style="cyan", no_wrap=True)
    table.add_column("Article", justify="right", style="yellow")
    table.add_column("Method", justify="right", style="blue")
    table.add_column("Status", justify="right", style="green")

    for i, flavor in enumerate(FLAVORS.keys()):
        for j, art in enumerate(articles.keys()):
            for k, method in enumerate(classifiers.keys()):
                dataset_name = f'Article {art} - {flavor}'
                status = exp_results.get(dataset_name, {}).get('methods', {}).get(method, None)
                status = '[green]DONE' if status else None
                table.add_row(flavor if j == 0 and k == 0 else None, art if k == 0 else None, method, status)

    print(table)

    for flavor, features_file in FLAVORS.items():
        print(Panel(f'[bold yellow] Cross-Validation Flavor {flavor.upper()}'), justify="center")

        print(Markdown("- **Prepare dataset**"))
        dataset_path = Path(outcomes_path) / features_file
        X = load_dataset(dataset_path)
        X = pd.DataFrame(X)
        print(TAB + '> Load the dataset [green][DONE]')

        if flavor != 'Bag-of-Words':
            # Remove '0:'
            to_drop = [e for e in X.columns if e.startswith('0:')]
            X.drop(columns=to_drop, inplace=True)
            print(TAB + '> Drop unecessary columns [green][DONE]')

        
        print(Markdown(f"- **Cross-Validate**"))
        for art, art_id in outcome_to_id.items():
            if f'{art_id}:1' in outcome_matrix.columns:
                print(Markdown(TAB + f'Article {art}'))
                y_art = outcome_matrix[(outcome_matrix[f'{art_id}:1'] == 1) | (outcome_matrix[f'{art_id}:0'] == 1)]
                y_art['decision'] = y_art.apply(lambda x: map_outcome(art_id, x), axis=1)
                print(TAB + '> Map outcomes for binary classification [green][DONE]')
                X_art = X[X.index.isin(y_art.index)]
                print(TAB + '> Filter dataset to keep only relevant cases [green][DONE]')
                y_art = y_art['decision']

                o = {'name': f'Article {art} - {flavor}'}
                dataset_name = o['name']


                if dataset_name not in exp_results:
                    exp_results[dataset_name] = {}

                #format_filter_output(dataset_name, o)
                update_dataset_result(dataset_name, o, result_file)
                update_dataset_filter_result(dataset_name, o, result_file)

                metadata = exp_results.get(dataset_name, {}).get('filter', {})
                metadata['size'] = metadata.get('size', int(y_art.shape[0]))
                metadata['violation'] = metadata.get('violation', int(y_art.sum()))
                metadata['non_violation'] = metadata['size'] - metadata['violation']
                metadata['prevalence'] = float(metadata['violation']) / metadata['size']
                
                update_dataset_metadata(dataset_name, metadata, result_file)
                print(TAB + '> Generate dataset metadata [green][DONE]')

                if flavor == 'Descriptive only':
                    update_article_desc(art, metadata, BINARY_DESC_OUTPUT_FILE)
                    print(TAB + '> Update dataset description [green][DONE]')

                
                for classifier_name, classifier in classifiers.items():
                    print(TAB + f'> [bold]{classifier_name}')
                    if exp_results.get(dataset_name, {}).get('methods', {}).get(classifier_name, None):
                        print(TAB + ' ⮡ Cross-Validation results already exist. [green][SKIP]')
                    else:
                        try:
                            scoring = make_scorers()
                            cv = TimeSeriesSplit(n_splits=k_fold) if as_time_series \
                                else StratifiedKFold(n_splits=k_fold)
                            scores = cross_validate(classifier, X_art, y_art,
                                cv=cv, 
                                scoring=scoring, 
                                return_train_score=True,
                                verbose=10,
                                n_jobs=-1, error_score='raise')
                            classifier_output = process_score(scores, scoring, seed)
                            update_classifier_result(
                                dataset_name, 
                                classifier_name, 
                                classifier_output, 
                                result_file
                            )
                            pass
                        except Exception as e:
                            print(e)



def main(args):
    console = Console(record=True)
    run(console, args.build, args.title, args.doc_ids, args.force)


def parse_args(parser):
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate datasets from a specific ECHR-OD build')
    parser.add_argument('--build', type=str, default="./build/echr_database/")
    parser.add_argument('--title', type=str)
    parser.add_argument('--doc_ids', type=str, default=None, nargs='+')
    parser.add_argument('-f', '--force', action='store_true')
    args = parse_args(parser)

    main(args)
