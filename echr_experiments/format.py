import numpy as np

from config import ROUND_DIGITS

def format_filter_output(name, output, format='md', file=None):
    if format == 'md':
        if not file:
            print('=' * 20)
            print('# Dataset {}'.format(name))
            print('=' * 20)
            print('## Filter dataset')
            for k,v in output.iteritems():
                print('\t{}: {}'.format(k.title(), v))


def format_method_output(method_name, classifier_output, format='md', file=None):
    print('-' * 20)
    print('## Method {}'.format(method_name))
    print('-' * 20)
    for k, v in classifier_output['train'].iteritems():
        print('\t{}={}'.format(k,  np.round_(v, ROUND_DIGITS)))
    print
    for k, v in classifier_output['test'].iteritems():
        print('\t{}={}'.format(k, np.round_(v, ROUND_DIGITS)))
    print
    for k, v in classifier_output['time'].iteritems():
        print('\t{}={}'.format(k, np.round_(v, ROUND_DIGITS)))
