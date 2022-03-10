import json
import numpy as np
import copy
from collections import OrderedDict
from echr_experiments.config import ROUND_DIGITS, MULTICLASS_OUTPUT_FILE
from echr_experiments.format import sort_article, data_to_method, FLAVORS_SHORT_FORM
from echr_experiments.utils import save

output_path= MULTICLASS_OUTPUT_FILE
dataset_short = FLAVORS_SHORT_FORM

def generate_latex_table_multiclass(_data, key=("acc", "Accuracy"), std=True, order=max):
    data = data_to_method(copy.deepcopy(_data))

    best_per_dataset = {}
    for dataset, methods in _data.items():
        name = dataset.split(' - ')[-1]
        best_per_dataset[name] = order([np.round_(m['test']['test_{}'.format(key[0])], ROUND_DIGITS) for m in methods['methods'].values()])
    nb_columns = 4 #
    column_placement = '|l' * (nb_columns) + '|'
    latex_output  = "\\begin{tabular}{" + column_placement + " }\n"
    latex_output += "\\hline\n"
    latex_output += " &  \multicolumn{3}{c|}{ " + key[1] + " - Multiclass} \\\\\n"
    latex_output += "\cline{2-4} & desc & BoW & both \\\\ \hline" + "\n"
    average = 0.
    max_m = max([len(m) for m in data.keys()])
    for i, method in enumerate(sorted(data.keys())):
        latex_output += '{message:<{fill}}'.format(message=method, fill=max_m)
        #for dataset in sorted(data[method].keys(), key=dataset_short.keys().index):
        for dataset in dataset_short.keys():
            if dataset in data[method]['Multiclass']:
                d = data[method]['Multiclass'][dataset]
                val = np.round_(d['test']['test_{}'.format(key[0])], ROUND_DIGITS)
                if val == best_per_dataset[dataset]:
                    latex_output += ' & {\\bf ' + '{:.4f}'.format(val) + '}'
                else:
                    latex_output += ' & {:.4f}'.format(val)
                if std:
                    latex_output += ' ({:.2f})'.format(np.round_(d['test']['test_{}_std'.format(key[0])], 2))
            else:
                latex_output += ' & missing '
        latex_output += '\\\\\n'
    latex_output += "\\hline\n"
    latex_output += "\end{tabular}"
    return latex_output

def main():
    with open(output_path) as f:
        data = json.load(f)

    keys = [
        ('acc', 'Accuracy'), 
        ('mcc', "MCC"), 
        ('precision', "Precision"), 
        ('recall', "Recall"), 
        ('f1_weighted', "F1 score"), 
        ('balanced_acc', "Balanced accuracy")
    ]
    for key in keys:
        save('multiclass_{}.tex'.format(key[0]), generate_latex_table_multiclass(data, key=key, std=True))

if __name__ == "__main__":
    # execute only if run as a script
    main()