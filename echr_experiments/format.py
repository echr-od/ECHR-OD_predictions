from collections import OrderedDict
import json
import numpy as np
import sys
from echr_experiments.config import ROUND_DIGITS, INPUT_PATH, OUTPUT_PATH

def format_filter_output(name, output, format='md', file=None):
    if format == 'md':
        if not file:
            print('=' * 20)
            print('# Dataset {}'.format(name))
            print('=' * 20)
            print('## Filter dataset')
            for k,v in output.items():
                print('\t{}: {}'.format(k.title(), v))


def format_method_output(method_name, classifier_output, format='md', file=None):
    print('-' * 20)
    print('## Method {}'.format(method_name))
    print('-' * 20)
    for k, v in classifier_output['train'].items():
        print('\t{}={}'.format(k,  np.round_(v, ROUND_DIGITS)))
    print
    for k, v in classifier_output['test'].items():
        print('\t{}={}'.format(k, np.round_(v, ROUND_DIGITS)))
    print
    for k, v in classifier_output['time'].items():
        print('\t{}={}'.format(k, np.round_(v, ROUND_DIGITS)))


def sort_article(a):
    i = a.split(' ')[-1]
    if i == 'p1-1':
        return sys.maxsize
    return int(i)


def number_cases_per_article(articles):
    res = {}
    for article in articles:
        #name = article.replace(' ', '_').lower()
        name = article.split()[-1]
        path = '{}/desc_binary.json'.format(OUTPUT_PATH, name)
        with open(path) as f:
            stats = json.load(f)
            res[article] = stats[name]['size']
    return res


def data_to_method(_data):
    data = {}
    for entry in _data.keys():
        d = entry.split(' - ')
        article = d[0]
        dataset = d[1]
        if "methods" in _data[entry]:
            for method, d in _data[entry]["methods"].items():
                if method not in data:
                    data[method] = {}
                if article not in data[method]:
                    data[method][article] = {}
                data[method][article][dataset] = d
    return data


def data_to_article(_data):
    data = {}
    prev = {}
    meta = {}
    for entry in _data.keys():
        d = entry.split(' - ')
        article = d[0]
        dataset = d[1]
        if article not in data:
            data[article] = {}
        if article != 'Multilabel' and article not in prev:
            prev[article] = _data[entry]['filter']['prevalence']
            meta[article]= _data[entry]['filter']
        if "methods" in _data[entry]:
            for method, d in _data[entry]["methods"].items():
                if method not in data[article]:
                    data[article][method] = {}
                data[article][method][dataset] = d
    return data, prev, meta


def floatify(X):
    Xf = []
    for x in X:
        Xf.append([float(e) for e in x])
    return Xf


FLAVORS_SHORT_FORM = OrderedDict([
    ("Descriptive only", "desc"),
    ("Bag-of-Words only", "BoW"),
    ("Descriptive and Bag-of-Words", "both")
])

FLAVORS_NAME_TO_FILE = OrderedDict([
    ("Descriptive only", "descriptive.txt"),
    ("Bag-of-Words only", "BoW.txt"),
    ("Descriptive and Bag-of-Words", "descriptive+BoW.txt")
])