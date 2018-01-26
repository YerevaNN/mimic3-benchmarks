import sklearn.utils as sk_utils
from mimic3models import metrics
import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', type=str)
    parser.add_argument('--test_listfile', type=str, default='../data/phenotyping/test/listfile.csv')
    parser.add_argument('--n_iters', type=int, default=10000)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.prediction, index_col=False)
    test_df = pd.read_csv(args.test_listfile, index_col=False)

    n_tasks = 25
    labels_cols = ["label_{}".format(i) for i in range(1, n_tasks + 1)]
    test_df.columns[2:] = labels_cols

    df = test_df.merge(pred_df, left_on='stay', right_on='stay', how='left', suffixes=['_l', '_r'])
    assert (df['prediction'].isnull().sum() == 0)
    assert (df['y_true_l'].equals(df['y_true_r']))
    assert (df['los'] == df['length of stay'])

    n_samples = df.shape[0]
    data = np.zeros((n_samples, 50))
    for i in range(1, n_tasks + 1):
        data[:, i] = df['pred_{}'.format(i)]
        data[:, 25 + i] = df['label_{}_l'.format(i)]

    ave_auc_macro = metrics.print_metrics_multilabel(data[:, 25:], data[:, :25], verbose=0)["ave_auc_macro"]
    aucs = []
    for i in range(args.n_iters):
        cur_data = sk_utils.resample(data, n_samples=len(data))
        ret = metrics.print_metrics_multilabel(cur_data[:, 25:], cur_data[:, :25], verbose=0)["ave_auc_macro"]
        aucs += [ret]

    print "{} iterations".format(args.n_iters)
    print "ave_auc_macro= {}".format(ave_auc_macro)
    print "mean = {}".format(np.mean(aucs))
    print "median = {}".format(np.median(aucs))
    print "std = {}".format(np.std(aucs))
    print "2.5% percentile = {}".format(np.percentile(aucs, 2.5))
    print "97.5% percentile = {}".format(np.percentile(aucs, 97.5))
