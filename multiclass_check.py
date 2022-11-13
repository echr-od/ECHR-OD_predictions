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
from bidict import bidict
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

    outcomes_path = 'data/input/datasets/'
    raw_outcome_file = Path(outcomes_path) / 'outcomes.txt'
    outcome_matrix_file = Path(outcomes_path) / 'outcomes_matrix.csv'

    print(Markdown("- **Prepare outcome matrix**"))
    OUTCOME_TO_ID =  'data/input/datasets/outcomes_variables.json'
    with open(OUTCOME_TO_ID, 'r') as f:
        outcome_to_id = json.load(f)

    
    if True: #not os.path.isfile(outcome_matrix_file) or force:
        print(TAB + '> Generate the outcome matrix [green][DONE]')
        outcomes_matrix = generate_outcomes_data(raw_outcome_file, outcome_to_id, filter_threshold=100)
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

    count = outcomes_matrix['article'].value_counts()
    count = json.loads(count.to_json())
    
    count = {k:v for k,v in count.items() if v > 100}
    articles_to_keep = list(count.keys())

    outcomes_matrix = outcomes_matrix[outcomes_matrix['article'].isin(articles_to_keep)]
    le = LabelEncoder()
    le.fit(outcomes_matrix[0])
    mapping_class = dict(zip(le.classes_, le.transform(le.classes_)))
    
    c_outcomes = bidict(outcome_to_id)
    metadata = json.loads(outcomes_matrix['article'].value_counts().to_json())
    metadata = {k:{'Size': v, 'Article': c_outcomes.inverse[int(k)] } for k,v in metadata.items()} # c_outcomes.inverse[int(k)]
    for art, mdata in metadata.items():
        mdata['Violation'] = outcomes_matrix[outcomes_matrix[0] == f'{art}:1'].shape[0]
        mdata['No-Violation'] = outcomes_matrix[outcomes_matrix[0] == f'{art}:0'].shape[0]
        mdata['Prevalence'] = mdata['Violation'] / (mdata['Violation'] + mdata['No-Violation'])

    update_article_desc('Multiclass', metadata, MULTICLASS_DESC_OUTPUT_FILE)
    print(TAB + '> Update dataset description [green][DONE]')

    print(Markdown("- **Experiment summary**"))
    FLAVORS = {'Descriptive only': 'descriptive.txt', 'Bag-of-Words only': 'BoW.txt', 'Descriptive and Bag-of-Words': 'descriptive+BoW.txt'}
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
                status = status_check('multiclass', exp_results, dataset_name, method)
                table.add_row(flavor if k == 0 else None, method, status)

    print(table)


def status_check(type_, exp_results, dataset_name, method): 
    schema = {
        'multiclass': 
            {
                "seed": 123456,
                "train": {
                    "train_balanced_acc": 0.46337330683830047,
                    "train_balanced_acc_std": 0.004365058332880957,
                    "train_acc": 0.7976210246013864,
                    "train_acc_std": 0.0025501452966668722,
                    "train_mcc": 0.7632405014241274,
                    "train_mcc_std": 0.0030062223243052773,
                    "train_kappa": 0.7582140960048799,
                    "train_kappa_std": 0.0031952208985870957,
                    "train_f1_weighted": 0.7653960248152075,
                    "train_f1_weighted_std": 0.0030502477712712086,
                    "train_precision": 0.8248963968512113,
                    "train_precision_std": 0.001836825104754893,
                    "train_recall": 0.7976210246013864,
                    "train_recall_std": 0.0025501452966668722,
                    "train_neg_log_loss": -3.1617222965864675,
                    "train_neg_log_loss_std": 0.0489818756884841,
                    "train_cm": 0.0,
                    "train_cm_std": 0.0
                },
                "test": {
                    "test_balanced_acc": 0.3121772289104234,
                    "test_balanced_acc_std": 0.0201282200126901,
                    "test_acc": 0.6692233225118418,
                    "test_acc_std": 0.019860500617787765,
                    "test_mcc": 0.6088265439051581,
                    "test_mcc_std": 0.023998185175912456,
                    "test_kappa": 0.5995785241028874,
                    "test_kappa_std": 0.02570737119830183,
                    "test_f1_weighted": 0.6123088415165113,
                    "test_f1_weighted_std": 0.023052429847943267,
                    "test_precision": 0.6145417798744217,
                    "test_precision_std": 0.02792737388843334,
                    "test_recall": 0.6692233225118418,
                    "test_recall_std": 0.019860500617787765,
                    "test_neg_log_loss": -5.8866167127920095,
                    "test_neg_log_loss_std": 0.30427812098778245,
                    "test_cm": 0.0,
                    "test_cm_std": 0.0
                },
                "time": {
                    "fit_time": 45.32742142677307,
                    "score_time": 9.886531734466553
                },
                "confusion_matrix": {}
        }
    }

    dataset_res = exp_results.get(dataset_name, {})
    if dataset_res is None:
        return '[blue]NOT STARTED'
    methods_res = dataset_res.get('methods', {})
    if methods_res is None:
        return '[blue]NOT STARTED'
    method_res = methods_res.get(method, None)
    if method_res is None:
        return '[blue]NOT STARTED'

    for k, v in schema[type_].items():
        if k not in method_res:
            return '[red]UNFINISHED'

        if isinstance(v, dict):
            for kk in v.keys():
                if kk not in method_res[k]:
                    return '[red]UNFINISHED'
    return '[green]DONE'


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
