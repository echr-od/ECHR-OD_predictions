import numpy as np
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
                                    AS_TIME_SERIES
from echr_experiments.format import format_filter_output, format_method_output
from echr_experiments.data import load_ECHR_instance, generate_datasets_descriptors
from echr_experiments.scorers import make_scorers, process_score
from echr_experiments.utils import update_classifier_result, \
                                   update_dataset_filter_result, \
                                   update_dataset_result

seed = SEED #random.randint(0,10000)
result_file = BINARY_OUTPUT_FILE
articles = BINARY_ARTICLES
flavors = BINARY_FLAVORS
as_time_series = AS_TIME_SERIES
k_fold = K_FOLD
feature_threshold = DEFAULT_FEATURE_THRESHOLD
classifiers = BINARY_CLASSIFIERS
datasets = generate_datasets_descriptors(
    articles=articles, 
    flavors_filter=flavors, 
    min_threshold=feature_threshold
)

for dataset in datasets:
    # Load datasets
    dataset_name = dataset.get('name', dataset['features'])
    X, y, o = load_ECHR_instance(
            X_file=dataset['features'], 
            y_file=dataset['labels'],
            min_threshold=dataset.get('min_threshold', feature_threshold)
        )
    try:
        prev = len([e for e in y if e == 1]) / (1. * len(y))
        o['prevalence'] = np.round_(prev, ROUND_DIGITS)
    except Exception as e:
        print(e)
        o['prevalence'] = 1.
    format_filter_output(dataset_name, o)
    update_dataset_result(dataset_name, dataset, result_file)
    update_dataset_filter_result(dataset_name, o, result_file)
    for classifier_name, classifier in classifiers.iteritems():
        try:
            scoring = make_scorers()
            cv = TimeSeriesSplit(n_splits=k_fold) if as_time_series \
                else StratifiedKFold(n_splits=k_fold, random_state=seed)
            scores = cross_validate(classifier, X, y, 
                cv=cv, 
                scoring=scoring, 
                return_train_score=True,
                verbose=10,
                n_jobs=-1)
            classifier_output = process_score(scores, scoring, seed)

            format_method_output(classifier_name, classifier_output)
            update_classifier_result(
                dataset_name, 
                classifier_name, 
                classifier_output, 
                result_file
            )
        except Exception as e:
            print(e)
 