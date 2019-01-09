import numpy as np
import json
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit

from echr_experiments.config import ROUND_DIGITS, \
                                    SEED, \
                                    BINARY_OUTPUT_FILE, \
                                    BINARY_ARTICLES, \
                                    BINARY_FLAVORS, \
                                    K_FOLD, \
                                    BINARY_CLASSIFIERS, \
                                    DEFAULT_FEATURE_THRESHOLD, \
                                    ANALYSIS_PATH, \
                                    INPUT_PATH
from echr_experiments.data import load_ECHR_instance, generate_datasets_descriptors
from echr_experiments.format import FLAVORS_NAME_TO_FILE
from echr_experiments.plot import plot_learning_curve
from echr_experiments.utils import get_best_configurations

seed = SEED
result_file = BINARY_OUTPUT_FILE
articles = BINARY_ARTICLES
flavors = BINARY_FLAVORS
k_fold = K_FOLD
feature_threshold = DEFAULT_FEATURE_THRESHOLD
classifiers = BINARY_CLASSIFIERS
analysis_path = ANALYSIS_PATH

def main():
    configurations = get_best_configurations(result_file)
    for c in configurations:
        print('# {} on {}'.format(c[1], c[0]))
        d = c[0].split(' - ')
        article = d[0].lower().replace(' ', '_')
        flavor = FLAVORS_NAME_TO_FILE[d[1]]
        dataset = {
            'name': c[0],
            'features': os.path.join(INPUT_PATH, article, flavor), 
            'labels': os.path.join(INPUT_PATH, article, 'outcomes.txt'), 
            'min_threshold': feature_threshold
        }
        method = classifiers[c[1]]
        X, y, o = load_ECHR_instance(
            X_file=dataset['features'], 
            y_file=dataset['labels'],
            min_threshold=dataset['min_threshold']
        )

        title = "{} on {}".format(c[1], c[0])
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
        print_range = (0.8, 1.01)
        if article == 'article_34':
            print_range = (0.3, 0.85)
        plt = plot_learning_curve(method, title, X, y, print_range, 
            cv=cv, 
            n_jobs=-1, 
            train_sizes=np.linspace(.1, 0.5, 5)
        )
        plt.savefig(os.path.join(analysis_path, 'lc', 'lc_{}_{}.png'.format(c[1], c[0])), dpi=600)
        plt.close('all')


if __name__ == "__main__":
    main()