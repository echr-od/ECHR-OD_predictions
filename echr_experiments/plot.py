import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.colors import ListedColormap
import numpy as np
import itertools
import pandas as pd


def make_permutations_histogram_plot(permutation_scores, score, prevalence):
    plt.hist(permutation_scores, 20, label='Permutation scores',
         edgecolor='black')
    ylim = plt.ylim()
    # BUG: vlines(..., linestyle='--') fails on older versions of matplotlib
    # plt.vlines(score, ylim[0], ylim[1], linestyle='--',
    #          color='g', linewidth=3, label='Classification Score'
    #          ' (pvalue %s)' % pvalue)
    # plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
    #          color='k', linewidth=3, label='Luck')
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
             label='Classification Score'
             ' (pvalue %s)' % pvalue)
    plt.plot(2 * [prevalence], ylim, '--k', linewidth=3, label='Luck')

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    return plt


def make_confusion_matrix_plot(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=sns.cubehelix_palette(as_cmap=True)):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = classes, columns = classes, )
    sns.set(font_scale=1.)
    mask = df_cm == 0.
    labels = []
    for x in cm:
        labels.append(['{:.2f}'.format(e) if e > 0.009999 else ' ' for e in x])
    labels = pd.DataFrame(labels, index = classes, columns = classes, )
    with sns.axes_style("white"):
        h = sns.heatmap(df_cm, annot=labels, annot_kws={"size": 5}, cmap = cmap, fmt='', xticklabels=classes, yticklabels=classes, mask=mask)# font size
    h.tick_params(labelsize=10)
    plt.xlabel('x-axis = predicted labels; y-axis = true labels')
    plt.subplots_adjust(left=0.17, bottom=0.21)
    ax = plt.axes()
    ax.set_title(title)
    return plt


def make_ROC_curve_plot(roc_auc):
    '''
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    '''
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.01, 0.1, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt