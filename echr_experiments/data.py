from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from config import ROUND_DIGITS


def check_empty_cases(X):
    nb_empty = 0
    for i, x in enumerate(X):
        m = [e for e in x if e == 0.0]
        if len(m) == len(x):
            nb_empty += 1
    return nb_empty


def load_ECHR_instance(X_file, y_file, min_threshold, multilabel=False, multiclass=False):
    with open(X_file) as file:
        data = file.readlines()
        data = [d.split() for d in data]
        for i, d in enumerate(data):
            data[i] = [int(''.join(e.split(':'))) for e in d]
        file.close()
    data = np.array(data, dtype=object)
    A = Counter(x for xs in data for x in set(xs))
    feature_count = {x : A[x] for x in A if A[x] > min_threshold }
    feature_mask = feature_count.keys()

    filter_output = {
        'threshold': int(min_threshold),
        'before': len(A),
        'after': len(feature_count),
        'ratio': np.round_(len(feature_count) / (1. * len(A)), ROUND_DIGITS)
    }
    data = [[f for f in x if f in feature_count] for x in data]

    docs = data
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in docs:
        for term in d:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    X = sparse.csr_matrix((data, indices, indptr), dtype=int)
    
    with open(y_file) as file:
        outcomes = file.readlines()
        if not multilabel and not multiclass:
            outcomes = np.array([int(d.split()[-1].split(':')[-1]) for d in outcomes])
        else:
            if multilabel:
                outcomes = [d.split()[1:] for d in outcomes]
                outcomes = MultiLabelBinarizer().fit_transform(outcomes)
            else:
                outcomes = np.array([d.split()[-1] for d in outcomes])
        file.close()
    y = outcomes
    return X, y, filter_output
    return X.toarray(), y, filter_output


def load_T_ECHR_instance(X_file, y_file, min_threshold):
    with open(X_file) as file:
        data = file.readlines()
        data = [d.split() for d in data]
        indices = [None] * len(data)
        for i, d in enumerate(data):
            indices[i] = [e.split(':')[0] for e in d]
            data[i] = [e.split(':') for e in d]
        file.close()
    data = np.array(data, dtype=object)
    A = Counter(x for xs in indices for x in set(xs))
    feature_count = {x : A[x] for x in A if A[x] > min_threshold }
    feature_mask = feature_count.keys()

    filter_output = {
        'threshold': int(min_threshold),
        'before': len(A),
        'after': len(feature_count),
        'ratio': np.round_(len(feature_count) / (1. * len(A)), ROUND_DIGITS)
    }
    data = [[f for f in x if f[0] in feature_count] for x in data]

    docs = data
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in docs:
        for term in d:
            index = vocabulary.setdefault(term[0], len(vocabulary))
            indices.append(index)
            data.append(term[1])
        indptr.append(len(indices))
    X = sparse.csr_matrix((data, indices, indptr), dtype=float)
    
    with open(y_file) as file:
        outcomes = file.readlines()
        outcomes = np.array([int(d.split()[-1].split(':')[-1]) for d in outcomes])
        file.close()
    y = outcomes
    return X.toarray(), y, filter_output


def load_CSV_ECHR_instance(X_file, y_file, min_threshold):
    df = pd.read_csv(X_file, header=None)
    X = df._get_numeric_data().as_matrix()
    df = pd.read_csv(y_file, header=None)
    y = [1 if e =='v' else 0 for e in df[1]]
    filter_output = {
        #'threshold': int(min_threshold),
        #'before': len(A),
        #'after': len(feature_count),
        #'ratio': np.round_(len(feature_count) / (1. * len(A)), ROUND_DIGITS)
    }
    return X, y, filter_output


def generate_datasets_descriptors(articles=None, flavors_filter=None, min_threshold=0):
    import itertools
    if articles is None:
        articles = map(lambda x: 'article_{}'.format(x), ['1', '2', '3', '5', '6', '8', '10', '11', '13', '34', 'p1'])
    flavors = [
        ('BoW', 'Bag-of-Words only'), 
        ('descriptive', 'Descriptive features only'),
        ('descriptive+BoW', 'Descriptive features and Bag-of-Words'), 
    ]
    if flavors_filter is not None:
        flavors = [f for f in flavors if f[0] in flavors_filter]
    keys = ['name', 'features', 'labels', 'min_threshold']
    datasets = []
    for i in itertools.product(articles, flavors):
        name = i[0] if not i[0].startswith('article') else 'Article {}'.format(i[0].split('_')[-1])
        base_path = 'data/input/{}/'.format(i[0])
        descriptor = {
            'name': '{} - {}'.format(name, i[1][1]),
            'features': base_path + i[1][0] + '.txt',
            'labels': base_path + 'outcomes.txt',
            'min_threshold': min_threshold
        }
        datasets.append(descriptor)
    return datasets