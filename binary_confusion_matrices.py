import json
from os import path
import numpy as np
import matplotlib.pyplot as plt
from echr_experiments.config import ROUND_DIGITS, ANALYSIS_PATH, BINARY_OUTPUT_FILE
from echr_experiments.format import data_to_article, floatify
from echr_experiments.plot import make_binary_confusion_matrix_plot
from echr_experiments.utils import get_best_configurations

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

output_file = BINARY_OUTPUT_FILE

def main():
    with open(output_file) as f:
        raw_results = json.load(f)

    configurations = get_best_configurations(output_file)
    for c in configurations:
        dataset = c[0]
        method = c[1]
        print('# {} on {}'.format(method, dataset))
        if "methods" in raw_results[dataset]:
                print('\t-{}'.format(method))
                for norm in [True, False]:
                    target = "test"
                    dd = raw_results[dataset]['methods'][method][target]
                    cnf_matrix = [
                        [dd["{}_tp{}".format(target, '_n' if norm else '')], dd["{}_fn{}".format(target, '_n' if norm else '')]],
                        [dd["{}_fp{}".format(target, '_n' if norm else '')], dd["{}_tn{}".format(target, '_n' if norm else '')]],
                    ]
                    class_names = ['Violation', 'No-violation']
                    cnf_matrix = floatify(cnf_matrix)
                    cnf_matrix = np .array([np.array(xi) for xi in cnf_matrix])
                    # Plot non-normalized confusion matrix
                    plt.figure()
                    title = '{}\n{}'.format(dataset, method)
                    make_binary_confusion_matrix_plot(cnf_matrix, classes=class_names, title=title)
                    title = 'binary_cm{}_{}_{}.png'.format('_normalized' if norm else '', target, dataset.split(' - ')[0].replace(' ', '_').lower())
                    plt.savefig(path.join(ANALYSIS_PATH, 'cm', title), dpi=600)
                    plt.close('all')

if __name__ == "__main__":
    main()