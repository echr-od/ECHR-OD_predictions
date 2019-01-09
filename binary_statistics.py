from scipy import stats

import json
from pprint import pprint
import numpy as np
import pandas as pd
import copy
import os
from collections import OrderedDict
from echr_experiments.config import ROUND_DIGITS
from echr_experiments.format import sort_article
from echr_experiments.utils import save

RESULT_PATH = 'data/output/result_binary.json'

dataset_short = OrderedDict([
    ("Descriptive features only", "desc"),
    ("Bag-of-Words only", "BoW"),
    ("Descriptive features and Bag-of-Words", "both")
])

def data_to_article(_data):
    data = {}
    prev = {}
    for entry in _data.keys():
        d = entry.split(' - ')
        article = d[0]
        dataset = d[1]
        if article not in data:
            data[article] = {}
        if article not in prev:
            prev[article] = _data[entry]['filter']['prevalence']
        if "methods" in _data[entry]:
            for method, d in _data[entry]["methods"].iteritems():
                if method not in data[article]:
                    data[article][method] = {}
                data[article][method][dataset] = d
    return data, prev

def generate_samples_per_article(name, _data, key="acc", order=max):
    data = copy.deepcopy(_data)

    best_per_dataset = {}
    for method, datasets in _data.iteritems():
        for dataset, res in datasets.iteritems():
            if dataset not in best_per_dataset:
                best_per_dataset[dataset] = np.round_(res['test']['test_{}'.format(key)], 4)
            else:
                best_per_dataset[dataset] = order(best_per_dataset[dataset], np.round_(res['test']['test_{}'.format(key)], 4))

    average = 0.
    sample_bow = []
    sample_bow_desc = []
    max_m = max([len(m) for m in data.keys()])
    for i, method in enumerate(sorted(data.keys())):
        for i, dataset in enumerate(dataset_short.keys()[1:]):
            if dataset in data[method]:
                d = data[method][dataset]
                val = np.round_(d['test']['test_{}'.format(key)], 4)
                if i == 1:
                    sample_bow.append(val)
                else:
                    sample_bow_desc.append(val)
    #print('BoW only')
    print('\n'.join(map(str, sample_bow)))
    print('\n')
    #print('BoW + desc')
    print('\n'.join(map( str,sample_bow_desc)))


def generate_samples_per_method(name, _data, key="acc", std=True, order=max):
    sample_bow = {}
    sample_bow_desc = {}
    data_per_article, prev = data_to_article(_data)
    for article, d in data_per_article.iteritems():
        data = copy.deepcopy(d)
        best_per_dataset = {}
        for method, datasets in data.iteritems():
            for dataset, res in datasets.iteritems():
                if dataset not in best_per_dataset:
                    best_per_dataset[dataset] = np.round_(res['test']['test_{}'.format(key)], 4)
                else:
                    best_per_dataset[dataset] = order(best_per_dataset[dataset], np.round_(res['test']['test_{}'.format(key)], 4))

        average = 0.
        for i, method in enumerate(sorted(data.keys())):
            if method not in sample_bow:
                sample_bow[method] = []
            if method not in sample_bow_desc:
                sample_bow_desc[method] = []
            for i, dataset in enumerate(dataset_short.keys()[1:]):
                if dataset in data[method]:
                    d = data[method][dataset]
                    val = np.round_(d['test']['test_{}'.format(key)], 4)
                    if i == 1:
                        sample_bow[method].append(val)
                    else:
                        sample_bow_desc[method].append(val)

    for k in sample_bow.keys():
        print('# {}'.format(k))
        #print('BoW only')
        print('\n'.join(map(str, sample_bow[k])))
        print('\n')
        #print('BoW + desc')
        print('\n'.join(map( str,sample_bow_desc[k])))
        print('\n')
        print(stats.wilcoxon(sample_bow[k], sample_bow_desc[k]))


def main():
    with open(RESULT_PATH) as f:
        _data = json.load(f)

    
    """
        RESULTS PER ARTICLE
    """
    data_per_article, prev = data_to_article(_data)
    keys = ['acc'] #, 'mcc', 'precision', 'recall', 'f1_weighted'] #, 'balanced_acc']
    for article, data in data_per_article.iteritems():
        for key in keys:
            print(' # {} - {}'.format(article, key))
            generate_samples_per_article(article, data, key=key, order=max)
            print('\n')

    keys = [
        ('acc', 'Accuracy'), 
        #('mcc', "Matthew Correlation Coefficient"), 
        #('precision', "Precision"), 
        #('recall', "Recall"), 
        #('f1_weighted', "F1 score"), 
        #('balanced_acc', "Balanced accuracy")
    ]
    for key in keys:
        generate_samples_per_method(key[1], _data, key[0], order=max)
    

if __name__ == "__main__":
    main()