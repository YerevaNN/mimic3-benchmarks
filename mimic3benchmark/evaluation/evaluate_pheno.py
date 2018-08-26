from __future__ import absolute_import
from __future__ import print_function

from mimic3models.metrics import print_metrics_multilabel, print_metrics_binary
import sklearn.utils as sk_utils
import numpy as np
import pandas as pd
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', type=str)
    parser.add_argument('--test_listfile', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/test/listfile.csv'))
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--save_file', type=str, default='pheno_results.json')
    args = parser.parse_args()

    pred_df = pd.read_csv(args.prediction, index_col=False, dtype={'period_length': np.float32})
    test_df = pd.read_csv(args.test_listfile, index_col=False, dtype={'period_length': np.float32})

    n_tasks = 25
    labels_cols = ["label_{}".format(i) for i in range(1, n_tasks + 1)]
    test_df.columns = list(test_df.columns[:2]) + labels_cols

    df = test_df.merge(pred_df, left_on='stay', right_on='stay', how='left', suffixes=['_l', '_r'])
    assert (df['pred_1'].isnull().sum() == 0)
    assert (df['period_length_l'].equals(df['period_length_r']))
    for i in range(1, n_tasks + 1):
        assert (df['label_{}_l'.format(i)].equals(df['label_{}_r'.format(i)]))

    metrics = [('Macro ROC AUC', 'ave_auc_macro'),
               ('Micro ROC AUC', 'ave_auc_micro'),
               ('Weighted ROC AUC', 'ave_auc_weighted')]

    data = np.zeros((df.shape[0], 50))
    for i in range(1, n_tasks + 1):
        data[:, i - 1] = df['pred_{}'.format(i)]
        data[:, 25 + i - 1] = df['label_{}_l'.format(i)]

    results = dict()
    results['n_iters'] = args.n_iters
    ret = print_metrics_multilabel(data[:, 25:], data[:, :25], verbose=0)
    for (m, k) in metrics:
        results[m] = dict()
        results[m]['value'] = ret[k]
        results[m]['runs'] = []

    for i in range(1, n_tasks + 1):
        m = 'ROC AUC of task {}'.format(i)
        results[m] = dict()
        results[m]['value'] = print_metrics_binary(data[:, 25 + i - 1], data[:, i - 1], verbose=0)['auroc']
        results[m]['runs'] = []

    for iteration in range(args.n_iters):
        cur_data = sk_utils.resample(data, n_samples=len(data))
        ret = print_metrics_multilabel(cur_data[:, 25:], cur_data[:, :25], verbose=0)
        for (m, k) in metrics:
            results[m]['runs'].append(ret[k])
        for i in range(1, n_tasks + 1):
            m = 'ROC AUC of task {}'.format(i)
            cur_auc = print_metrics_binary(cur_data[:, 25 + i - 1], cur_data[:, i - 1], verbose=0)['auroc']
            results[m]['runs'].append(cur_auc)

    reported_metrics = [m for m, k in metrics]
    reported_metrics += ['ROC AUC of task {}'.format(i) for i in range(1, n_tasks + 1)]

    for m in reported_metrics:
        runs = results[m]['runs']
        results[m]['mean'] = np.mean(runs)
        results[m]['median'] = np.median(runs)
        results[m]['std'] = np.std(runs)
        results[m]['2.5% percentile'] = np.percentile(runs, 2.5)
        results[m]['97.5% percentile'] = np.percentile(runs, 97.5)
        del results[m]['runs']

    print("Saving the results (including task specific metrics) in {} ...".format(args.save_file))
    with open(args.save_file, 'w') as f:
        json.dump(results, f)

    print("Printing the summary of results (task specific metrics are skipped) ...")
    for i in range(1, n_tasks + 1):
        m = 'ROC AUC of task {}'.format(i)
        del results[m]
    print(results)


if __name__ == "__main__":
    main()
