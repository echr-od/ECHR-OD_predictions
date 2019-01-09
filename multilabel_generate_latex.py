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
    for method, datasets in _data.iteritems():
        for dataset, res in datasets.iteritems():
            if dataset not in best_per_dataset:
                best_per_dataset[dataset] = np.round_(res['test']['test_{}'.format(key)], 4)
            else:
                best_per_dataset[dataset] = order(best_per_dataset[dataset], np.round_(res['test']['test_{}'.format(key)], 4))

    nb_columns = 4 #
    column_placement = '|l' * (nb_columns) + '|'
    latex_output  = "\\begin{tabular}{" + column_placement + " }\n"
    latex_output += "\\hline\n"
    latex_output += " &  \multicolumn{3}{c|}{" + name + "} \\\\\n"
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

def generate_latex_table_binary_best_article(metric_name, _data, key="acc", std=True):
    data = copy.deepcopy(_data)
    best_per_article = {}
    for article, entry in data.iteritems():
        for method, datasets in entry.iteritems():
            for dataset, res in datasets.iteritems():
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
    for article, entry in best_per_article.iteritems():
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
    for method, entry in data.iteritems():
        if method not in best_per_method:
            best_per_method[method] = {}
        for article, datasets in entry.iteritems():
            for dataset, res in datasets.iteritems():
                val = float(res['test']['test_{}'.format(key)])
                if article not in best_per_method[method]:
                    best_per_method[method][article] = res
                else:
                    if val > float(best_per_method[method][article]['test']['test_{}'.format(key)]):
                        best_per_method[method][article] = res

    average_per_method = {}
    micro_average_per_method = {}
    for method, entry in best_per_method.iteritems():
        average = 0.
        micro_average = 0.
        total_cases = 0.
        cases_per_articles = number_cases_per_article(entry.keys())
        for article, _ in entry.iteritems():
            average += best_per_method[method][article]['test']['test_{}'.format(key)]
            micro_average += best_per_method[method][article]['test']['test_{}'.format(key)] * float(cases_per_articles[article])
            total_cases += float(cases_per_articles[article])
        average_per_method[method] = average / len(entry)
        micro_average_per_method[method] = micro_average / total_cases

    def sort_method(m):
        return float(average_per_method[m])

    average = 0.
    for method, entry in average_per_method.iteritems():
        average += float(entry)
    average /= len(average_per_method.keys())

    micro_average = 0.
    for method, entry in micro_average_per_method.iteritems():
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
    latex_output += 'Average & {:.4f} & {:.4f} & \\\\\n'.format(np.round_(average, 4), np.round_(micro_average, 4))
    latex_output += "\\hline\n"
    latex_output += "\end{tabular}"
    return latex_output


def main():
    with open(result_path) as f:
        _data = json.load(f)

    """
        RESULTS PER ARTICLE
    """
    data_per_article, prev = data_to_article(_data)
    keys = ['acc', 'precision', 'recall', 'f1_weighted', 'zero_one_loss', 'jaccard_similarity_score', 'hamming_loss'] #, 'balanced_acc']
    for article, data in data_per_article.iteritems():
        for key in keys:
            std = False if key != 'acc' else True
            save('{}_{}.tex'.format(article.replace(' ', '_').lower(), key), 
                generate_latex_table_multilabel_article(article, data, 
                    key=key, 
                    std=std, 
                    order=max)
                )


if __name__ == "__main__":
    main()