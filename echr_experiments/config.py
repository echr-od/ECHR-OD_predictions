from os import path
from collections import OrderedDict

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, \
							 AdaBoostClassifier, \
							 BaggingClassifier, \
							 ExtraTreesClassifier, \
							 GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, TimeSeriesSplit, StratifiedKFold

# Experiment replication
SEED = 123456
K_FOLD = 10
AS_TIME_SERIES = False

# Analysis and format
ROUND_DIGITS = 4

# General settings
ANALYSIS_PATH = 'data/analysis'
OUTPUT_PATH = 'data/output'
INPUT_PATH = 'data/input'
DEFAULT_FEATURE_THRESHOLD = 0

# ECHR specific
BINARY_OUTPUT_FILE = path.join(OUTPUT_PATH, 'result_binary.json')
BINARY_DESC_OUTPUT_FILE = path.join(OUTPUT_PATH, 'desc_binary.json')

BINARY_ARTICLES = map(lambda x: 'article_{}'.format(x), 
	['1', '2', '3', '5', '6', '8', '10', '11', '13', '34', 'p1']
)
BINARY_FLAVORS = ['BoW', 'descriptive', 'descriptive+BoW']
BINARY_CLASSIFIERS = OrderedDict({
    ### Naive Bayes
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Multinomial Naive Bayes": MultinomialNB(),
    # CANNOT BE USED WITH SPARSE MATRIX # "Gaussian Naive Bayes": GaussianNB(),

    ### K-Neighbors
    "K-Neighbors": KNeighborsClassifier(),

    #### SVM
    "Linear SVC": CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=StratifiedKFold(n_splits=K_FOLD)), #SVC(kernel='linear', probability=False),
    #"Linear SVC": SVC(kernel='linear', probability=True),
    "RBF SVC": SVC(probability=True),

    # CANNOT BE USED WITH SPARSE MATRIX "Gaussian Process": GaussianProcessClassifier(),

    ### Tree-based
    "Extra Tree": ExtraTreeClassifier(max_depth=None, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "BaggingClassifier": BaggingClassifier(random_state=SEED),
    "Ensemble Extra Tree": ExtraTreesClassifier(n_estimators=100, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    "AdaBoost": AdaBoostClassifier(random_state=SEED),

    ### Others
    "Neural Net": MLPClassifier(max_iter=500, random_state=SEED),
    # CANNOT BE USED WITH SPARSE MATRIX "QDA": QuadraticDiscriminantAnalysis()
})

MULTICLASS_OUTPUT_FILE = path.join(OUTPUT_PATH, 'result_multiclass.json')
MULTICLASS_DESC_OUTPUT_FILE = path.join(OUTPUT_PATH, 'desc_multiclass.json')
MULTICLASS_ARTICLES = ['multiclass']
MULTICLASS_FLAVORS = ['BoW', 'descriptive', 'descriptive+BoW']
MULTICLASS_CLASSIFIERS = OrderedDict({
    ### Naive Bayes
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Multinomial Naive Bayes": MultinomialNB(),
    # CANNOT BE USED WITH SPARSE MATRIX # "Gaussian Naive Bayes": GaussianNB(),

    #### SVM
    "Linear SVC": CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=StratifiedKFold(n_splits=K_FOLD)),

    # CANNOT BE USED WITH SPARSE MATRIX "Gaussian Process": GaussianProcessClassifier(),

    ### Tree-based
    "Extra Tree": ExtraTreeClassifier(max_depth=None, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "BaggingClassifier": BaggingClassifier(random_state=SEED),
    "Ensemble Extra Tree": ExtraTreesClassifier(n_estimators=100, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    "AdaBoost": AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), random_state=SEED),

    ### Others
    "Neural Net": MLPClassifier(random_state=SEED),
    # CANNOT BE USED WITH SPARSE MATRIX "QDA": QuadraticDiscriminantAnalysis()
})

MULTILABEL_OUTPUT_FILE = path.join(OUTPUT_PATH, 'result_multilabel.json')
MULTILABEL_DESC_OUTPUT_FILE = path.join(OUTPUT_PATH, 'desc_multilabel.json')
MULTILABEL_ARTICLES = ['multilabel']
MULTILABEL_FLAVORS = ['BoW', 'descriptive', 'descriptive+BoW']
MULTILABEL_CLASSIFIERS = OrderedDict({
    "Extra Tree": ExtraTreeClassifier(max_depth=None),
    "Decision Tree": DecisionTreeClassifier(max_depth=None),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Ensemble Extra Tree": ExtraTreesClassifier(n_estimators=100),
    "Neural Net": MLPClassifier(max_iter=500),
})
