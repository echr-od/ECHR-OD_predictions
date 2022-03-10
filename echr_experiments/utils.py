import json
from os import path
from echr_experiments.config import ANALYSIS_PATH
from echr_experiments.format import data_to_article


def update_article_desc(article, result, path):
    data = {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except:
        pass

    if article not in data:
        data[article] = {}

    data[article] = result

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_dataset_metadata(dataset_name, result, path):
    with open(path, "r") as f:
        data = json.load(f)

    if 'filter' not in data[dataset_name]:
        data[dataset_name]['filter'] = {}

    data[dataset_name]['filter'] = result

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_classifier_result(dataset_name, classifier_name, result, path):
    with open(path, "r") as f:
        data = json.load(f)

    if 'methods' not in data[dataset_name]:
        data[dataset_name]['methods'] = {}

    data[dataset_name]['methods'][classifier_name] = result

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_dataset_filter_result(dataset_name, result, path):
    with open(path, "r") as f:
        data = json.load(f)
    
    if 'filter' not in data[dataset_name]:
        data[dataset_name]['filter'] = {}

    data[dataset_name]['filter'] = result

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_dataset_result(dataset_name, result, path):
    data = {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except:
        pass

    if dataset_name not in data:
        data[dataset_name] = {}

    if 'descriptor' not in data[dataset_name]:
        data[dataset_name]['descriptor'] = {}
    
    data[dataset_name]['descriptor'] = result

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def get_best_configurations(path):
    with open(path) as f:
        data = json.load(f)

    data, prev, meta = data_to_article(data)
    key = 'acc'
    best_per_article = {}
    for article, entry in data.items():
        for method, datasets in entry.items():
            for dataset, res in datasets.items():
                val = float(res['test']['test_{}'.format(key)])
                if article not in best_per_article:
                    best_per_article[article] = res
                    best_per_article[article]['flavor'] = dataset
                    best_per_article[article]['method'] = method
                else:
                    if val > float(best_per_article[article]['test']['test_{}'.format(key)]):
                        best_per_article[article] = res
                        best_per_article[article]['flavor'] = dataset
                        best_per_article[article]['method'] = method
    res = []
    for article, element in best_per_article.items():
        dataset_name = '{} - {}'.format(article, element['flavor'])
        res.append([dataset_name, element['method']])
    return res

def save(filename, data):
    with open(path.join(ANALYSIS_PATH, 'tables', filename), 'w') as f:  
        f.write(data)