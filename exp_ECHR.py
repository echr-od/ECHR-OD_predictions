import time
import random
from collections import OrderedDict
import numpy as np
import json
from sklearn.model_selection import cross_validate, TimeSeriesSplit, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from echr_experiments.config import ROUND_DIGITS
from echr_experiments.format import format_filter_output, format_method_output
from echr_experiments.data import load_ECHR_instance, generate_datasets_descriptors
from echr_experiments.scorers import make_scorers, process_score
from echr_experiments.utils import update_classifier_result, update_dataset_filter_result, update_dataset_result

from sklearn.metrics import roc_curve, auc

seed = 123456 #random.randint(0,10000)
result_file = 'data/output/result.json'
as_time_series = False
training_size = 0.9
k_fold = 10

classifiers = OrderedDict({
    #"Bernoulli Naive Bayes": BernoulliNB(),
    #"Multinomial Naive Bayes": MultinomialNB(),
    #"Gaussian Naive Bayes": GaussianNB(),
    #"K-Neighbors": KNeighborsClassifier(n_jobs=-1),
    #"Linear SVC": SVC(kernel='linear', probability=True),
    #"RBF SVC": SVC(probability=True),
    #"RBF NuSVC": NuSVC(probability=True),
    #"Gaussian Process": GaussianProcessClassifier(),
    #"Extra Tree": ExtraTreeClassifier(),
    #"Decision Tree": DecisionTreeClassifier(max_depth=None),
    #"Random Forest": RandomForestClassifier(n_estimators=100),
    #"BaggingClassifier": BaggingClassifier(),
    #"Ensemble Extra Tree": ExtraTreesClassifier(n_estimators=100),
    #"Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    #"Neural Net": MLPClassifier(max_iter=500),
    "QDA": QuadraticDiscriminantAnalysis()
})

datasets = [
    # EXAMPLE OF DATASET DESCRIPTION
    {
        'name': 'Article 8 - Bag-of-Words only',
        'features': 'data/input/article_8/BoW.txt', 
        'labels': 'data/input/article_8/outcomes.txt', 
        'min_treshold': 100
    }
]

datasets = generate_datasets_descriptors()

for dataset in datasets:
    # Load datasets
    dataset_name = dataset.get('name', dataset['features'])
    X, y, o = load_ECHR_instance(
            X_file=dataset['features'], 
            y_file=dataset['labels'],
            min_threshold=dataset.get('min_threshold', 100)
        )
    prev = len([e for e in y if e == 1]) / (1. * len(y))
    o['prevalence'] = np.round_(prev, ROUND_DIGITS)
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
                n_jobs=-1)
            classifier_output = process_score(scores, scoring, seed)
            format_method_output(classifier_name, classifier_output)
            update_classifier_result(dataset_name, classifier_name, classifier_output, result_file)
        except Exception as e:
            print(e)
    