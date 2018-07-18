from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, \
                            confusion_matrix, \
                            cohen_kappa_score, \
                            fbeta_score, \
                            brier_score_loss, \
                            precision_recall_curve, \
                            accuracy_score, \
                            matthews_corrcoef
import numpy as np

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

def make_scorers():
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

    return scoring

def process_score(scores, scoring, seed):
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
    test_keys = map(lambda x: 'test_{}'.format(x), scoring_keys)
    for k in test_keys:
        classifier_output['test'][k] = mean_scores[k]
    time_keys = ['fit_time', 'score_time']
    for k in time_keys:
        classifier_output['time'][k] = mean_scores[k]
    classifier_output['test']['test_acc_std'] = scores['test_acc'].std()
    classifier_output['train']['train_acc_std'] = scores['train_acc'].std()
    return classifier_output