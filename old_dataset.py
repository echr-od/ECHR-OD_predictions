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
from echr_experiments.data import load_CSV_ECHR_instance, generate_datasets_descriptors
from echr_experiments.scorers import make_scorers, process_score
from echr_experiments.utils import update_classifier_result, update_dataset_filter_result, update_dataset_result

from sklearn.metrics import roc_curve, auc

seed = 123456 #random.randint(0,10000)
result_file = 'data/output/result_old.json'
as_time_series = False
training_size = 0.9
k_fold = 10

classifiers = OrderedDict({
    #"Bernoulli Naive Bayes": BernoulliNB(),
    #"Multinomial Naive Bayes": MultinomialNB(),
    #"Gaussian Naive Bayes": GaussianNB(),
    #"K-Neighbors": KNeighborsClassifier(n_jobs=-1)
    #"Linear SVC": SVC(kernel='linear', probability=True)#,
    #"RBF SVC": SVC(probability=True),
    #"RBF NuSVC": NuSVC(probability=True),
    #"Gaussian Process": GaussianProcessClassifier(),
    #Extra Tree": ExtraTreeClassifier(),
    #"Decision Tree": DecisionTreeClassifier(max_depth=None),
    #"Random Forest": RandomForestClassifier(n_estimators=100),
    #"BaggingClassifier": BaggingClassifier(),
    #"Ensemble Extra Tree": ExtraTreesClassifier(n_estimators=100),
    #"Gradient Boosting": GradientBoostingClassifier(),
    #"AdaBoost": AdaBoostClassifier(),
    #"Neural Net": MLPClassifier(max_iter=500)
    #"QDA": QuadraticDiscriminantAnalysis()
})


datasets = [
    {
        'name': 'Article 3 - Full',
        'features': 'data/input/old_dataset/Article3/ngrams_a3_full.csv',
        'labels': 'data/input/old_dataset/Article3/cases_a3.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 3 - Topics',
        'features': 'data/input/old_dataset/Article3/topics3.csv',
        'labels': 'data/input/old_dataset/Article3/cases_a3.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 3 - Circumstances',
        'features': 'data/input/old_dataset/Article3/ngrams_a3_circumstances.csv',
        'labels': 'data/input/old_dataset/Article3/cases_a3.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 3 - Law',
        'features': 'data/input/old_dataset/Article3/ngrams_a3_law.csv',
        'labels': 'data/input/old_dataset/Article3/cases_a3.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 3 - Procedure',
        'features': 'data/input/old_dataset/Article3/ngrams_a3_procedure.csv',
        'labels': 'data/input/old_dataset/Article3/cases_a3.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 3 - Relevant Law',
        'features': 'data/input/old_dataset/Article3/ngrams_a3_relevantLaw.csv',
        'labels': 'data/input/old_dataset/Article3/cases_a3.csv', 
        'min_treshold': 100
    },{
        'name': 'Article 6 - Full',
        'features': 'data/input/old_dataset/Article6/ngrams_a6_full.csv',
        'labels': 'data/input/old_dataset/Article6/cases_a6.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 6 - Topics',
        'features': 'data/input/old_dataset/Article6/topics6.csv',
        'labels': 'data/input/old_dataset/Article6/cases_a6.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 6 - Circumstances',
        'features': 'data/input/old_dataset/Article6/ngrams_a6_circumstances.csv',
        'labels': 'data/input/old_dataset/Article6/cases_a6.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 6 - Law',
        'features': 'data/input/old_dataset/Article6/ngrams_a6_law.csv',
        'labels': 'data/input/old_dataset/Article6/cases_a6.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 6 - Procedure',
        'features': 'data/input/old_dataset/Article6/ngrams_a6_procedure.csv',
        'labels': 'data/input/old_dataset/Article6/cases_a6.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 6 - Relevant Law',
        'features': 'data/input/old_dataset/Article6/ngrams_a6_relevantLaw.csv',
        'labels': 'data/input/old_dataset/Article6/cases_a6.csv', 
        'min_treshold': 100
    },    {
        'name': 'Article 8 - Full',
        'features': 'data/input/old_dataset/Article8/ngrams_a8_full.csv',
        'labels': 'data/input/old_dataset/Article8/cases_a8.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 8 - Topics',
        'features': 'data/input/old_dataset/Article8/topics8.csv',
        'labels': 'data/input/old_dataset/Article8/cases_a8.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 8 - Circumstances',
        'features': 'data/input/old_dataset/Article8/ngrams_a8_circumstances.csv',
        'labels': 'data/input/old_dataset/Article8/cases_a8.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 8 - Law',
        'features': 'data/input/old_dataset/Article8/ngrams_a8_law.csv',
        'labels': 'data/input/old_dataset/Article8/cases_a8.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 8 - Procedure',
        'features': 'data/input/old_dataset/Article8/ngrams_a8_procedure.csv',
        'labels': 'data/input/old_dataset/Article8/cases_a8.csv', 
        'min_treshold': 100
    },
    {
        'name': 'Article 8 - Relevant Law',
        'features': 'data/input/old_dataset/Article8/ngrams_a8_relevantLaw.csv',
        'labels': 'data/input/old_dataset/Article8/cases_a8.csv', 
        'min_treshold': 100
    },
]

for dataset in datasets:
    # Load datasets
    dataset_name = dataset.get('name', dataset['features'])
    X, y, o = load_CSV_ECHR_instance(
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
