import json
import numpy as np
import copy
import os
from collections import OrderedDict
from echr_experiments.config import ROUND_DIGITS, BINARY_OUTPUT_FILE
from echr_experiments.format import sort_article, \
                                    number_cases_per_article, \
                                    data_to_method, \
                                    data_to_article, \
                                    FLAVORS_SHORT_FORM
from echr_experiments.utils import save

result_path = BINARY_OUTPUT_FILE
flavors_short_names = FLAVORS_SHORT_FORM

def generate_latex_table_binary_article(name, _data, key=("acc", "Accuracy"), std=True, order=max, prev=None):
    data = copy.deepcopy(_data)

    best_per_dataset = {}
    for method, datasets in _data.items():
        for dataset, res in datasets.items():
            if dataset not in best_per_dataset:
                best_per_dataset[dataset] = np.round_(res['test']['test_{}'.format(key[0])], 4)
            else:
                best_per_dataset[dataset] = order(best_per_dataset[dataset], np.round_(res['test']['test_{}'.format(key[0])], 4))

    nb_columns = 4 #
    column_placement = '|l' * (nb_columns) + '|'
    latex_output  = "\\begin{tabular}{" + column_placement + " }\n"
    latex_output += "\\hline\n"
    if prev is not None:
        latex_output += 'Prev=' + str(round(prev, 4)) + " &  \multicolumn{3}{c|}{" + key[1] + ' - ' + name + "} \\\\\n"
    else:
        latex_output += " &  \multicolumn{3}{c|}{" + name + "} \\\\\n"
    latex_output += "\cline{2-4} & desc & BoW & both \\\\ \hline" + "\n"
    average = 0.
    max_m = max([len(m) for m in data.keys()])
    for i, method in enumerate(sorted(data.keys())):
        latex_output += '{message:<{fill}}'.format(message=method, fill=max_m)
        for dataset in flavors_short_names.keys():
            if dataset in data[method]:
                d = data[method][dataset]
                val = np.round_(d['test']['test_{}'.format(key[0])], 4)
                if val == best_per_dataset[dataset]:
                    latex_output += ' & {\\bf ' + '{:.4f}'.format(val) + '}'
                else:
                    latex_output += ' & {:.4f}'.format(val)
                if std:
                    latex_output += ' ({:.2f})'.format(np.round_(d['test']['test_{}_std'.format(key[0])], 2))
            else:
                latex_output += ' & missing'
        latex_output += '\\\\\n'
    latex_output += "\\hline\n"
    latex_output += "\end{tabular}"
    return latex_output

def generate_latex_table_binary_best_article(metric_name, _data, key="acc", std=True):
    data = copy.deepcopy(_data)
    best_per_article = {}
    for article, entry in data.items():
        for method, datasets in entry.items():
            for dataset, res in datasets.items():
                val = float(res['test']['test_{}'.format(key)])
                if article not in best_per_article:
                    best_per_article[article] = res
                    best_per_article[article]['flavor'] = dataset
                    best_per_article[article]['method'] = method
                else:
                    if val > float(best_per_article[article]['test']['test_{}'.format(key)]):
                        best_per_article[article] = res
                        best_per_article[article]['flavor'] = dataset
                        best_per_article[article]['method'] = method
    average = 0.
    micro_average = 0.
    total_cases = 0.
    cases_per_articles = number_cases_per_article(best_per_article.keys())
    for article, entry in best_per_article.items():
        average += float(entry['test']['test_{}'.format(key)])
        micro_average += float(entry['test']['test_{}'.format(key)]) * float(cases_per_articles[article])
        total_cases += float(cases_per_articles[article])
    average /= len(best_per_article.keys())
    micro_average /= total_cases

    nb_columns = 4 #
    column_placement = '|l' * (nb_columns) + '|'
    latex_output  = "\\begin{tabular}{" + column_placement + " }\n"
    latex_output += "\\hline\n"
    latex_output += "Article & " + metric_name + " & Method & Flavor \\\\ \hline\n"
    for i, article in enumerate(sorted(best_per_article.keys(), key=sort_article)):
        method = best_per_article[article]['method']
        flavor = best_per_article[article]['flavor']
        latex_output += '{}'.format(article)
        val = np.round_(best_per_article[article]['test']['test_{}'.format(key)], 4)
        latex_output += ' & {:.4f}'.format(val)
        if std:
            latex_output += ' ({:.2f})'.format(np.round_(best_per_article[article]['test']['test_{}_std'.format(key)], 2))
        latex_output += ' & ' + method + ' & ' + flavor + '\\\\\n'
    latex_output += 'Average & {:.4f} & & \\\\\n'.format(np.round_(average, 4))
    latex_output += 'Micro average & {:.4f} & & \\\\\n'.format(np.round_(micro_average, 4))
    latex_output += "\\hline\n"
    latex_output += "\end{tabular}"
    return latex_output

def generate_latex_table_binary_overall(metric_name, _data, key="acc", std=True, micro=True):
    data = copy.deepcopy(_data)
    best_per_method = {}
    for method, entry in data.items():
        if method not in best_per_method:
            best_per_method[method] = {}
        for article, datasets in entry.items():
            for dataset, res in datasets.items():
                val = float(res['test']['test_{}'.format(key)])
                if article not in best_per_method[method]:
                    best_per_method[method][article] = res
                else:
                    if val > float(best_per_method[method][article]['test']['test_{}'.format(key)]):
                        best_per_method[method][article] = res

    average_per_method = {}
    micro_average_per_method = {}
    for method, entry in best_per_method.items():
        average = 0.
        micro_average = 0.
        total_cases = 0.
        cases_per_articles = number_cases_per_article(entry.keys())
        for article, _ in entry.items():
            average += best_per_method[method][article]['test']['test_{}'.format(key)]
            micro_average += best_per_method[method][article]['test']['test_{}'.format(key)] * float(cases_per_articles[article])
            total_cases += float(cases_per_articles[article])
        average_per_method[method] = average / len(entry)
        micro_average_per_method[method] = micro_average / total_cases

    def sort_method(m):
        return float(average_per_method[m])

    average = 0.
    for method, entry in average_per_method.items():
        average += float(entry)
    average /= len(average_per_method.keys())

    micro_average = 0.
    for method, entry in micro_average_per_method.items():
        micro_average += float(entry)
    micro_average /= len(micro_average_per_method.keys())

    nb_columns = 4 if micro else 3
    max_m = max([len(m) for m in data.keys()])
    column_placement = '|l' * (nb_columns) + '|'
    latex_output  = "\\begin{tabular}{" + column_placement + " }\n"
    latex_output += "\\hline\n"
    latex_output += "{message:<{fill}} & ".format(message="Method", fill=max_m) + metric_name + "{} & Rank \\\\ \hline\n".format(' & Micro {}'.format(metric_name) if micro else '')
    for i, method in enumerate(sorted(average_per_method.keys(), key=sort_method, reverse=True)):
        latex_output += '{message:<{fill}}'.format(message=method, fill=max_m)
        val = np.round_(average_per_method[method], 4)
        latex_output += ' & {:.4f}'.format(val)
        #if std:
        #    latex_output += ' ({:.2f})'.format(np.round_(average_per_method[method], 2))
        if micro:
            latex_output += ' & {:.4f}'.format(micro_average_per_method[method])
        latex_output += ' & ' + str(i + 1) + '\\\\\n'
    if key == 'acc':
        latex_output += 'Average & {:.4f} & {:.4f} & \\\\\n'.format(np.round_(average, 4), np.round_(micro_average, 4))
    else:
        latex_output += 'Average & {:.4f} & \\\\\n'.format(np.round_(average, 4))
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

    keys = [
        ('acc', 'Accuracy'), 
        ('mcc', "MCC"), 
        ('precision', "Precision"), 
        ('recall', "Recall"), 
        ('f1_weighted', "F1 score"), 
        #('balanced_acc', "Balanced accuracy")
    ]
    for article, data in data_per_article.items():
        for key in keys:
            std = False if key[0] != 'acc' else True
            save('binary_{}_{}.tex'.format(key[0], article.replace(' ', '_').lower()), 
                generate_latex_table_binary_article(article, data, 
                    key=key, 
                    std=std, 
                    order=max,
                    prev=prev[article])
                )

    """
        BEST RESULT PER ARTICLE WITH FLAVOR AND METHOD
    """
    data_per_article, prev, meta = data_to_article(_data)
    keys = [
        ('acc', 'Accuracy'), 
        ('mcc', "MCC"), 
        ('precision', "Precision"), 
        ('recall', "Recall"), 
        ('f1_weighted', "F1 score"), 
        #('balanced_acc', "Balanced accuracy")
    ]
    for key in keys:
        save('binary_{}_best.tex'.format(key[0]),
                generate_latex_table_binary_best_article(key[1], data_per_article, 
                    key=key[0], 
                    std=False if key[0] != 'acc' else True)
        )

    """
        AVERAGE BEST RESULTS PER METHODS ON ALL ARTICLES
    """
    data_per_method = data_to_method(_data)
    keys = [
        ('acc', 'Accuracy'), 
        ('mcc', "MCC"), 
        ('precision', "Precision"), 
        ('recall', "Recall"), 
        ('f1_weighted', "F1 score"), 
        #('balanced_acc', "Balanced accuracy")
    ]
    for key in keys:
        save('binary_{}_summary.tex'.format(key[0]),
            generate_latex_table_binary_overall(key[1], data_per_method, 
                        key=key[0], 
                        std=False if key[0] != 'acc' else True,
                        micro=False if key[0] != 'acc' else True)
        )


if __name__ == "__main__":
    main()