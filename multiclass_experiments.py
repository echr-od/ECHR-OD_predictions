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
from scipy import sparse
from sklearn.model_selection import cross_validate, StratifiedKFold
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
                                    MULTICLASS_DESC_OUTPUT_FILE, \
                                    MULTICLASS_OUTPUT_FILE, \
                                    MULTICLASS_ARTICLES, \
                                    MULTICLASS_FLAVORS, \
                                    K_FOLD, \
                                    AS_TIME_SERIES, \
                                    MULTICLASS_CLASSIFIERS, \
                                    DEFAULT_FEATURE_THRESHOLD
from echr_experiments.format import format_filter_output, format_method_output
from echr_experiments.data import load_ECHR_instance, generate_datasets_descriptors
from echr_experiments.scorers import make_scorers, process_score, calculate_average_cm
from echr_experiments.utils import update_classifier_result, \
                                   update_dataset_filter_result, \
                                   update_dataset_metadata, \
                                   update_dataset_result, \
                                   update_article_desc

seed = SEED
result_file = MULTICLASS_OUTPUT_FILE
articles = MULTICLASS_ARTICLES
flavors = MULTICLASS_FLAVORS
k_fold = K_FOLD
as_time_series = AS_TIME_SERIES
feature_threshold = DEFAULT_FEATURE_THRESHOLD
classifiers = MULTICLASS_CLASSIFIERS



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

    #force = True

    print(Markdown("- **Step configuration**"))
    input_folder = os.path.join(build, 'raw', 'raw_cases_info')
    output_folder = path.join(build, 'intermediate')
    print(TAB + '> Step folder: {}'.format(path.join(build, 'cases_info')))
    make_build_folder(console, output_folder, force, strict=False)

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
        print(outcomes_matrix)
        outcomes_matrix.to_csv(outcome_matrix_file)
    else:
        print(TAB + '> Load the outcome matrix [green][DONE]')
    outcome_matrix = pd.read_csv(outcome_matrix_file)



    # Class encoding
    le = LabelEncoder()
    le.fit(outcomes_matrix[0])
    mapping_class = dict(zip(le.classes_, le.transform(le.classes_)))
    outcomes_matrix['decision'] = le.transform(outcomes_matrix[0])
    outcomes_matrix['article'] = outcomes_matrix[0].apply(lambda x: x.split(':')[0])
    CM = []  # Confusion matrices 

    print(outcomes_matrix)

    count = outcomes_matrix['article'].value_counts()
    count = json.loads(count.to_json())
    
    count = {k:v for k,v in count.items() if v > 100}
    articles_to_keep = list(count.keys())

    outcomes_matrix = outcomes_matrix[outcomes_matrix['article'].isin(articles_to_keep)]
    le = LabelEncoder()
    le.fit(outcomes_matrix[0])
    mapping_class = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(list(le.classes_))
    #exit(1)
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
                dataset_name = f'Multiclass - {flavor}'
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
        y = outcomes_matrix['decision']

        o = {'name': f'Multiclass - {flavor}'}
        dataset_name = o['name']


        if dataset_name not in exp_results:
            exp_results[dataset_name] = {}

        #format_filter_output(dataset_name, o)
        update_dataset_result(dataset_name, o, result_file)
        update_dataset_filter_result(dataset_name, o, result_file)

        metadata = exp_results.get(dataset_name, {}).get('filter', {})
        metadata['size'] = metadata.get('size', int(y.shape[0]))
        #metadata['violation'] = metadata.get('violation', int(y_art.sum()))
        #metadata['non_violation'] = metadata['size'] - metadata['violation']
        #metadata['prevalence'] = float(metadata['violation']) / metadata['size']
                
        update_dataset_metadata(dataset_name, metadata, result_file)
        print(TAB + '> Generate dataset metadata [green][DONE]')

        #if flavor == 'Descriptive only':
        #    update_article_desc(art, metadata, MULTICLASS_DESC_OUTPUT_FILE)
        #    print(TAB + '> Update dataset description [green][DONE]')

        CM = []
        for classifier_name, classifier in classifiers.items():
                    print(TAB + f'> [bold]{classifier_name}')
                    if exp_results.get(dataset_name, {}).get('methods', {}).get(classifier_name, None):
                        print(TAB + ' ⮡ Cross-Validation results already exist. [green][SKIP]')
                    else:
                        try:
                            scoring = make_scorers(multiclass=True, CM=CM)
                            cv = TimeSeriesSplit(n_splits=k_fold) if as_time_series \
                                else StratifiedKFold(n_splits=2) #k_fold) #, random_state=seed)
                            #scores = cross_validate(classifier, X_art.to_numpy(), y_art.to_numpy(),
                            scores = cross_validate(classifier, X, y,
                                cv=cv, 
                                scoring=scoring, 
                                return_train_score=True,
                                verbose=10,
                                n_jobs=None, error_score='raise')
                            classifier_output = process_score(scores, scoring, seed, multilabel=True)
                            cm = calculate_average_cm(CM, train_score=True)
                            classifier_output['confusion_matrix'] = cm
                            classifier_output['confusion_matrix']['class_labels'] = list(le.classes_) #list(outcomes_matrix[0].value_counts().index) #list(le.classes_)
                            #format_method_output(classifier_name, classifier_output)
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
    parser = argparse.ArgumentParser(description='Multiclass experiments')
    parser.add_argument('--build', type=str, default="./build/echr_database/")
    parser.add_argument('-f', '--force', action='store_true')
    args = parse_args(parser)

    main(args)
