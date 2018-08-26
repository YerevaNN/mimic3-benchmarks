from __future__ import absolute_import
from __future__ import print_function

from mimic3models.metrics import print_metrics_binary
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
                        default=os.path.join(os.path.dirname(__file__), '../../data/decompensation/test/listfile.csv'))
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--save_file', type=str, default='decomp_results.json')
    args = parser.parse_args()

    pred_df = pd.read_csv(args.prediction, index_col=False, dtype={'period_length': np.float32})
    test_df = pd.read_csv(args.test_listfile, index_col=False, dtype={'period_length': np.float32})

    df = test_df.merge(pred_df, left_on=['stay', 'period_length'], right_on=['stay', 'period_length'],
                       how='left', suffixes=['_l', '_r'])
    assert (df['prediction'].isnull().sum() == 0)
    assert (df['y_true_l'].equals(df['y_true_r']))

    metrics = [('AUC of ROC', 'auroc'),
               ('AUC of PRC', 'auprc'),
               ('min(+P, Se)', 'minpse')]

    data = np.zeros((df.shape[0], 2))
    data[:, 0] = np.array(df['prediction'])
    data[:, 1] = np.array(df['y_true_l'])

    results = dict()
    results['n_iters'] = args.n_iters
    ret = print_metrics_binary(data[:, 1], data[:, 0], verbose=0)
    for (m, k) in metrics:
        results[m] = dict()
        results[m]['value'] = ret[k]
        results[m]['runs'] = []

    for i in range(args.n_iters):
        cur_data = sk_utils.resample(data, n_samples=len(data))
        ret = print_metrics_binary(cur_data[:, 1], cur_data[:, 0], verbose=0)
        for (m, k) in metrics:
            results[m]['runs'].append(ret[k])

    for (m, k) in metrics:
        runs = results[m]['runs']
        results[m]['mean'] = np.mean(runs)
        results[m]['median'] = np.median(runs)
        results[m]['std'] = np.std(runs)
        results[m]['2.5% percentile'] = np.percentile(runs, 2.5)
        results[m]['97.5% percentile'] = np.percentile(runs, 97.5)
        del results[m]['runs']

    print("Saving the results in {} ...".format(args.save_file))
    with open(args.save_file, 'w') as f:
        json.dump(results, f)

    print(results)


if __name__ == "__main__":
        main()
