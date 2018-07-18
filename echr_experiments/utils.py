import json

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