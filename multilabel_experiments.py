#!/usr/bin/python
import argparse
import json
import os
from os import listdir, path
import copy
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bidict import bidict
from scipy import sparse
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

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
from echr_experiments.config import ROUND_DIGITS, \
                                    SEED, \
                                    MULTILABEL_DESC_OUTPUT_FILE, \
                                    MULTILABEL_OUTPUT_FILE, \
                                    MULTILABEL_ARTICLES, \
                                    MULTILABEL_FLAVORS, \
                                    K_FOLD, \
                                    AS_TIME_SERIES, \
                                    MULTILABEL_CLASSIFIERS, \
                                    DEFAULT_FEATURE_THRESHOLD, \
                                    ANALYSIS_PATH
from echr_experiments.format import format_filter_output, format_method_output
from echr_experiments.data import load_ECHR_instance, generate_datasets_descriptors
from echr_experiments.scorers import make_scorers, process_score, calculate_average_cm
from echr_experiments.utils import update_classifier_result, \
                                   update_dataset_filter_result, \
                                   update_dataset_metadata, \
                                   update_dataset_result, \
                                   update_article_desc

seed = SEED
result_file = MULTILABEL_OUTPUT_FILE
articles = MULTILABEL_ARTICLES
flavors = MULTILABEL_FLAVORS
k_fold = K_FOLD
as_time_series = AS_TIME_SERIES
feature_threshold = DEFAULT_FEATURE_THRESHOLD
classifiers = MULTILABEL_CLASSIFIERS



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
        outcomes[0] = outcomes[0].apply(lambda x: x[1])

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
            if f'{art}:0' not in yououtcomes.columns:
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

    return outcomes


def map_outcome(art, x):
            if x[f'{art}:1'] == 1:
                return 1 
            elif x[f'{art}:0'] == 1:
                return 0
            else: 
                return -1


def generate_outcomes_data(y_file, outcome_to_id, filter_threshold=100):
    # Generate hot-one outcome matrix
    with open(y_file) as file:
        f = lambda x: {x.split(':')[0]:x.split(':')[1]}
        outcomes = file.readlines()
        outcomes = pd.DataFrame(outcomes)
        outcomes[0] = outcomes[0].apply(lambda x: x.strip().split())
        outcomes['caseid'] = outcomes[0].apply(lambda x: x[0])
        outcomes[0] = outcomes[0].apply(lambda x: x[1:])

        '''
        # Generate hot-one outcome matrix
        mlb = MultiLabelBinarizer(sparse_output=True)
        outcomes = outcomes.join(
                    pd.DataFrame.sparse.from_spmatrix(
                        mlb.fit_transform(outcomes.pop(0)),
                        index=outcomes.index,
                        columns=mlb.classes_))
        '''
    '''
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
    '''
    return outcomes


def _generate_outcomes_data(y_file, outcome_to_id, filter_threshold=100):
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


    #outcomes.replace(['0', 0], np.nan, inplace=True)
    #for c in outcomes.columns:
    #    outcomes[c].loc[outcomes[c] == 0] = np.nan
    #outcomes.drop(columns=to_drop, inplace=True)
    #print(outcomes)
    #exit(1)
    #outcomes['decision'] = outcomes.notna().dot(outcomes.columns+',').str.rstrip(',')
    #outcomes['labels'] = outcomes.apply(lambda x: [outcomes.columns[i] for i, e in enumerate(x) if e == 1], axis=1)
    return outcomes


def load_dataset(X_file, min_threshold=0):
    with open(X_file) as file:
        f = lambda x: {x.split(':')[0]:x.split(':')[1]}
        X = file.readlines()
        X = pd.DataFrame(X)
        X[0] = X[0].apply(lambda x: sorted(x.strip().split()))
        #X['caseid'] = X[0].apply(lambda x: x[0])
        #X[0] = X[0].apply(lambda x: x[1:])



        # Generate hot-one outcome matrix
        mlb = MultiLabelBinarizer(sparse_output=True)
        X = X.join(
                    pd.DataFrame.sparse.from_spmatrix(
                        mlb.fit_transform(X.pop(0)),
                        index=X.index,
                        columns=mlb.classes_))
        return X


def run(console, build, force):
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

    
    if True: #not os.path.isfile(outcome_matrix_file) or force:
        print(TAB + '> Generate the outcome matrix [green][DONE]')
        outcomes_matrix = generate_outcomes_data(raw_outcome_file, outcome_to_id, filter_threshold=100)
        outcomes_matrix.to_csv(outcome_matrix_file)
    else:
        print(TAB + '> Load the outcome matrix [green][DONE]')
    outcomes_matrix = pd.read_csv(outcome_matrix_file)

    label_numbers = outcomes_matrix.apply(lambda x: len(x['0'].split()), axis=1)
    label_numbers = label_numbers.value_counts()
    fig = plt.figure()
    ax = label_numbers.plot.bar(x='lab', y='val', rot=0)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    fig.savefig(Path(ANALYSIS_PATH) / 'labels_per_cases.png')
    print(TAB + '> Generate the plot of labels counts [green][DONE]')

    outcomes_matrix = _generate_outcomes_data(raw_outcome_file, outcome_to_id, filter_threshold=100)
    
    metadata = outcomes_matrix[outcomes_matrix.columns[1:]]
    metadata = pd.DataFrame(metadata.sum())
    metadata['label'] = metadata.index
    metadata['article'] = metadata.apply(lambda x: x.label.split(':')[0], axis=1)
    summary = metadata[0].T

    arts = metadata['article'].unique()
    c_outcomes = bidict(outcome_to_id)
    metadata = {}
    for art in arts:
        if art not in metadata:
            metadata[art] = {}
        metadata[art]['Article'] = c_outcomes.inverse[int(art)]
        if f'{art}:1' in summary and f'{art}:0' in summary:
            metadata[art]['Size'] = int(summary[f'{art}:1'] + summary[f'{art}:0'])
        elif f'{art}:1' not in summary:
            metadata[art]['Size'] = int(summary[f'{art}:0'])
        else:
            metadata[art]['Size'] = int(summary[f'{art}:1'])

        if f'{art}:1' in summary:
            metadata[art]['Violation'] = int(summary[f'{art}:1'])
        else:
            metadata[art]['Violation'] = 0

        if f'{art}:0' in summary:
            metadata[art]['No-Violation'] = int(summary[f'{art}:0'])
        else:
            metadata[art]['No-Violation'] = 0
        metadata[art]['Prevalence'] = float(metadata[art]['Violation'] / (metadata[art]['Violation'] + metadata[art]['No-Violation']))

    update_article_desc('Multilabel', metadata, MULTILABEL_DESC_OUTPUT_FILE)
    print(TAB + '> Update dataset description [green][DONE]')
    
    CM = []  # Confusion matrices 
    print(Markdown("- **Experiment summary**"))
    FLAVORS = {'Descriptive only': 'descriptive.txt', 'Bag-of-Words only': 'BoW.txt', 'Descriptive and Bag-of-Words': 'descriptive+BoW.txt'}
    print(f"  | Flavors: {len(FLAVORS)}")
    print(f"  | Methods: {len(classifiers)}")
    print(f"  = {len(FLAVORS) * len(classifiers)} cross-validation procedures")
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
    table.add_column("Method", justify="right", style="blue")
    table.add_column("Status", justify="right", style="green")

    for i, flavor in enumerate(FLAVORS.keys()):
            for k, method in enumerate(classifiers.keys()):
                dataset_name = f'Multilabel - {flavor}'
                status = exp_results.get(dataset_name, {}).get('methods', {}).get(method, None)
                status = '[green]DONE' if status else None
                table.add_row(flavor if k == 0 else None, method, status)

    print(table)

    for flavor, features_file in FLAVORS.items():
        print(Panel(f'[bold yellow] Cross-Validation Flavor {flavor.upper()}'), justify="center")

        print(Markdown("- **Prepare dataset**"))
        dataset_path = Path(outcomes_path) / features_file
        X = load_dataset(dataset_path)
        X = pd.DataFrame(X)
        X = X[X.index.isin(outcomes_matrix.index)]
        print(TAB + '> Load the dataset [green][DONE]')

        if flavor != 'Bag-of-Words':
            # Remove '0:'
            to_drop = [e for e in X.columns if e.startswith('0:')]
            X.drop(columns=to_drop, inplace=True)
            print(TAB + '> Drop unecessary columns [green][DONE]')

        
        print(Markdown(f"- **Cross-Validate**"))
        y = outcomes_matrix.drop(columns='caseid').to_numpy()#['decision']

        o = {'name': f'Multilabel - {flavor}'}
        dataset_name = o['name']


        if dataset_name not in exp_results:
            exp_results[dataset_name] = {}

        update_dataset_result(dataset_name, o, result_file)
        update_dataset_filter_result(dataset_name, o, result_file)

        metadata = exp_results.get(dataset_name, {}).get('filter', {})
        metadata['size'] = metadata.get('size', int(y.shape[0]))        
        update_dataset_metadata(dataset_name, metadata, result_file)
        print(TAB + '> Generate dataset metadata [green][DONE]')

        CM = []
        for classifier_name, classifier in classifiers.items():
                    print(TAB + f'> [bold]{classifier_name}')
                    if exp_results.get(dataset_name, {}).get('methods', {}).get(classifier_name, None):
                        print(TAB + ' ⮡ Cross-Validation results already exist. [green][SKIP]')
                    else:
                        try:
                            scoring = make_scorers(multilabel=True, CM=CM)
                            cv = TimeSeriesSplit(n_splits=k_fold) if as_time_series \
                                else KFold(n_splits=k_fold) 
                            scores = cross_validate(classifier, X, y,
                                cv=cv, 
                                scoring=scoring, 
                                return_train_score=True,
                                verbose=10,
                                n_jobs=-1, error_score='raise')
                            classifier_output = process_score(scores, scoring, seed, multilabel=True)
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
    run(console, args.build, args.force)


def parse_args(parser):
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilabel experiments')
    parser.add_argument('--build', type=str, default="./build/echr_database/")
    parser.add_argument('-f', '--force', action='store_true')
    args = parse_args(parser)

    main(args)
