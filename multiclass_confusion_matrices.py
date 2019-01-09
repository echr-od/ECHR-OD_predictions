import json
from os import path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.colors import ListedColormap
from echr_experiments.config import ROUND_DIGITS, ANALYSIS_PATH, MULTICLASS_OUTPUT_FILE
from echr_experiments.format import floatify
from echr_experiments.plot import make_confusion_matrix_plot

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

output_file = MULTICLASS_OUTPUT_FILE

def main():
    with open(output_file) as f:
        raw_results = json.load(f)

    #dataset = "Multiclass - Descriptive features only"
    for dataset in raw_results.keys():
        if "methods" in raw_results[dataset]:
            for method, data in raw_results[dataset]["methods"].iteritems():
                target = "test"
                #method = "Random Forest"
                cnf_matrix = data["confusion_matrix"]["{}_n".format(target)]
                class_names = data["confusion_matrix"]['class_labels']
                class_names = ['Art. {} - {}'.format(n.split(':')[0], 'V' if n.split(':')[1] == '1' else 'NV') for n in class_names]
                cnf_matrix = floatify(cnf_matrix)
                cnf_matrix = np .array([np.array(xi) for xi in cnf_matrix])

                np.set_printoptions(precision=2)

                    # Plot non-normalized confusion matrix
                plt.figure()
                title = '{} on {}'.format(method, dataset.split(' - ')[-1])
                make_confusion_matrix_plot(cnf_matrix, classes=class_names, title=title)
                plt.savefig(path.join(ANALYSIS_PATH, 'cm', 'multiclass_cm_{}_{}_{}.png'.format(target, method.replace(' ', '_').lower(), dataset.split(' - ')[-1].replace(' ', '_').lower())), dpi=600)
                plt.close('all')

if __name__ == "__main__":
    # execute only if run as a script
    main()