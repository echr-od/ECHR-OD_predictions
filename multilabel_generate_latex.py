import json
import numpy as np
import copy
import os
from collections import OrderedDict
from echr_experiments.config import ROUND_DIGITS, MULTILABEL_OUTPUT_FILE
from echr_experiments.format import sort_article, \
                                    number_cases_per_article, \
                                    data_to_method, \
                                    data_to_article, \
                                    FLAVORS_SHORT_FORM
from echr_experiments.utils import save

result_path = MULTILABEL_OUTPUT_FILE
dataset_short = FLAVORS_SHORT_FORM

def generate_latex_table_multilabel_article(name, _data, key="acc", std=True, order=max):
    data = copy.deepcopy(_data)

    best_per_dataset = {}
    for method, datasets in _data.items():
        for dataset, res in datasets.items():
            if dataset not in best_per_dataset:
                best_per_dataset[dataset] = np.round_(res['test']['test_{}'.format(key)], 4)
            else:
                best_per_dataset[dataset] = order(best_per_dataset[dataset], np.round_(res['test']['test_{}'.format(key)], 4))

    nb_columns = 4 #
    column_placement = '|l' * (nb_columns) + '|'
    latex_output  = "\\begin{tabular}{" + column_placement + " }\n"
    latex_output += "\\hline\n"
    latex_output += " &  \multicolumn{3}{c|}{"+ key.replace('_', ' ').title() + " - " + name + "} \\\\\n"
    latex_output += "\cline{2-4} & desc & BoW & both \\\\ \hline" + "\n"
    average = 0.
    max_m = max([len(m) for m in data.keys()])
    for i, method in enumerate(sorted(data.keys())):
        latex_output += '{message:<{fill}}'.format(message=method, fill=max_m)
        for dataset in dataset_short.keys():
            if dataset in data[method]:
                d = data[method][dataset]
                val = np.round_(d['test']['test_{}'.format(key)], 4)
                if val == best_per_dataset[dataset]:
                    latex_output += ' & {\\bf ' + '{:.4f}'.format(val) + '}'
                else:
                    latex_output += ' & {:.4f}'.format(val)
                if std:
                    latex_output += ' ({:.2f})'.format(np.round_(d['test']['test_{}_std'.format(key)], 2))
            else:
                latex_output += ' & missing'
        latex_output += '\\\\\n'
    latex_output += "\\hline\n"
    latex_output += "\end{tabular}"
    return latex_output


def main():
    with open(result_path) as f:
        _data = json.load(f)

    """
        RESULTS PER ARTICLE
    """
    data_per_article, prev, meta = data_to_article(_data)
    keys = ['acc', 'precision', 'recall', 'f1_weighted', 'zero_one_loss', 'jaccard_similarity_score', 'hamming_loss'] #, 'balanced_acc']
    for article, data in data_per_article.items():
        for key in keys:
            std = False if key != 'acc' else True
            save('{}_{}.tex'.format(article.replace(' ', '_').lower(), key), 
                generate_latex_table_multilabel_article(article, data, 
                    key=key, 
                    std=std, 
                    order=max if key != 'hamming_loss' else min)
                )


if __name__ == "__main__":
    main()