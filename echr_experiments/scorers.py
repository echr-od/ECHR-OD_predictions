from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, \
                            confusion_matrix, \
                            cohen_kappa_score, \
                            fbeta_score, \
                            brier_score_loss, \
                            precision_recall_curve, \
                            accuracy_score, \
                            matthews_corrcoef, \
                            zero_one_loss, \
                            coverage_error, \
                            label_ranking_loss, \
                            hamming_loss, \
                            jaccard_similarity_score, \
                            balanced_accuracy_score
import numpy as np
from functools import partial
from echr_experiments.config import ROUND_DIGITS

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def tp_n(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm[0, 0]

def tn_n(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm[1, 1]

def fp_n(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm[1, 0]

def fn_n(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm[0, 1]

def make_scorers(multilabel=False, multiclass=False, CM=None):
    if not multilabel and not multiclass:
        scoring = {
            'acc': 'accuracy',
            'mcc': make_scorer(matthews_corrcoef),
            'kappa': make_scorer(cohen_kappa_score),
            'average_precision': 'average_precision',
            'roc_auc': 'roc_auc',
            'f1_weighted': 'f1_weighted',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'neg_log_loss': 'neg_log_loss',
            #'fbeta_score': make_scorer(fbeta_score),
            'brier_score_loss': make_scorer(brier_score_loss),
            'tp' : make_scorer(tp), 
            'tn' : make_scorer(tn),
            'fp' : make_scorer(fp), 
            'fn' : make_scorer(fn),
            'tp_n' : make_scorer(tp_n), 
            'tn_n' : make_scorer(tn_n),
            'fp_n' : make_scorer(fp_n), 
            'fn_n' : make_scorer(fn_n)
        }
    elif multilabel:
        scoring = {
            'acc': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'recall': 'recall_weighted',
            'zero_one_loss': make_scorer(zero_one_loss),
            'coverage_error': make_scorer(coverage_error),
            'label_ranking_loss': make_scorer(label_ranking_loss),
            'hamming_loss': make_scorer(hamming_loss),
            'jaccard_similarity_score': make_scorer(jaccard_similarity_score),
            'average_precision': 'average_precision',
            'precision': 'precision_weighted'
            #'roc_auc': 'roc_auc'
        }
    else:
        scoring = {
            'balanced_acc': make_scorer(balanced_accuracy_score),
            'acc': 'accuracy',
            'mcc': make_scorer(matthews_corrcoef),
            'kappa': make_scorer(cohen_kappa_score),
            'f1_weighted': 'f1_weighted',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'neg_log_loss': 'neg_log_loss',
        }
        f_cm = partial(add_cm, cm=CM)
        f_cm.__name__ = 'cm'
        scoring['cm'] = make_scorer(f_cm)

    return scoring

def add_cm(y_true, y_pred, cm=None):
    if cm is not None:
        cm.append(confusion_matrix(y_true, y_pred))
    return 0

def calculate_average_cm(CM, train_score=True):
    rounder = np.vectorize(lambda x: '{:.4f}'.format(np.round_(x, ROUND_DIGITS)))
    result = {}
    folds = len(CM) if not train_score else len(CM) / 2
    cm = CM[0::2]
    #print('FOLD: {}'.format(folds))
    #print('LEN {}'.format(len(cm)))
    avg_cm = np.sum([i / float(folds) for i in cm], 0)
    #print('AVG {}'.format(len(avg_cm)))
    if train_score:
        result['test'] =rounder(avg_cm).tolist()
        result['test_n'] = rounder(avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis]).tolist()
        result['train'] = np.sum([i / float(folds) for i in CM[1::2]], 0)
        result['train_n'] = rounder(result['train'].astype('float') / result['train'].sum(axis=1)[:, np.newaxis]).tolist()
        result['train'] = rounder(result['train']).tolist()
        '''
        print('TEST')
        acc = 0.
        t = 0.
        for i,x in enumerate(result['test']):
            acc += float(x[i])
            t += sum([float(e) for e in x])
            print(i, x[i], t)
        acc /= t
        print('ACC TEST {}'.format(acc))

        print('TRAIN')
        acc = 0.
        t = 0.
        for i,x in enumerate(result['train']):
            acc += float(x[i])
            t += sum([float(e) for e in x])
            print(i, x[i], t)
        acc /= t
        print('ACC TRAIN {}'.format(acc))
        '''
    else:
        result['test'] = rounder(avg_cm)
    return result


def process_score(scores, scoring, seed, multilabel=False):
    classifier_output = {
        'seed': seed,
        'train': {},
        'test': {},
        'time': {}
    }
    scoring_keys = scoring.keys()
    mean_scores = {k:v.mean() for k,v in scores.iteritems()}
    train_keys = map(lambda x: 'train_{}'.format(x), scoring_keys)
    for k in train_keys:
        classifier_output['train'][k] = mean_scores[k]
        if multilabel:
            classifier_output['train']['{}_std'.format(k)] = scores[k].std()
    test_keys = map(lambda x: 'test_{}'.format(x), scoring_keys)
    for k in test_keys:
        classifier_output['test'][k] = mean_scores[k]
        if multilabel:
            classifier_output['test']['{}_std'.format(k)] = scores[k].std()
    time_keys = ['fit_time', 'score_time']
    for k in time_keys:
        classifier_output['time'][k] = mean_scores[k]
    if not multilabel:
        classifier_output['test']['test_acc_std'] = scores['test_acc'].std()
        classifier_output['train']['train_acc_std'] = scores['train_acc'].std()

    return classifier_output