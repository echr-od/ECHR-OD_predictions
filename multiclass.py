import json
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from echr_experiments.config import ROUND_DIGITS, \
                                    SEED, \
                                    MULTICLASS_OUTPUT_FILE, \
                                    MULTICLASS_ARTICLES, \
                                    MULTICLASS_FLAVORS, \
                                    K_FOLD, \
                                    MULTICLASS_CLASSIFIERS, \
                                    DEFAULT_FEATURE_THRESHOLD
from echr_experiments.format import format_filter_output, format_method_output
from echr_experiments.data import load_ECHR_instance, generate_datasets_descriptors
from echr_experiments.scorers import make_scorers, process_score, calculate_average_cm
from echr_experiments.utils import update_classifier_result, \
                                   update_dataset_filter_result, \
                                   update_dataset_result

seed = SEED
result_file = MULTICLASS_OUTPUT_FILE
articles = MULTICLASS_ARTICLES
flavors = MULTICLASS_FLAVORS
k_fold = K_FOLD
feature_threshold = DEFAULT_FEATURE_THRESHOLD
classifiers = MULTICLASS_CLASSIFIERS
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
            min_threshold=dataset.get('min_threshold', 1),
            multilabel=False,
            multiclass=True
        )
    # Class encoding
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    CM = []  # Confusion matrices 

    format_filter_output(dataset_name, o)
    update_dataset_result(dataset_name, dataset, result_file)
    update_dataset_filter_result(dataset_name, o, result_file)
    for classifier_name, classifier in classifiers.iteritems():
        try:
            scoring = make_scorers(multiclass=True, CM=CM)
            cv = StratifiedKFold(n_splits=k_fold, random_state=seed)
            scores = cross_validate(classifier, X, y, 
                cv=cv, 
                scoring=scoring, 
                return_train_score=True,
                verbose=10)
                #, n_jobs=-1)  # Parallel execution cannot be used due to the CM
            classifier_output = process_score(scores, scoring, seed, multilabel=True)
            cm = calculate_average_cm(CM, train_score=True)
            classifier_output['confusion_matrix'] = cm
            classifier_output['confusion_matrix']['class_labels'] = list(le.classes_)

            format_method_output(classifier_name, classifier_output)
            update_classifier_result(
                dataset_name, 
                classifier_name, 
                classifier_output, 
                result_file
            )
        except Exception as e:
            print(e)
