import sklearn.utils as sk_utils
from mimic3models import metrics
import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', type=str)
    parser.add_argument('--test_listfile', type=str, default='../data/decompensation/test/listfile.csv')
    parser.add_argument('--n_iters', type=int, default=1000)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.prediction, index_col=False)
    test_df = pd.read_csv(args.test_listfile, index_col=False)

    df = test_df.merge(pred_df, left_on=['stay', 'period_length'], right_on=['stay', 'period_length'],
                       how='left', suffixes=['_l', '_r'])
    assert (df['prediction'].isnull().sum() == 0)
    assert (df['y_true_l'].equals(df['y_true_r']))

    n_samples = df.shape[0]
    data = np.zeros((n_samples, 2))
    data[:, 0] = np.array(df['prediction'])
    data[:, 1] = np.array(df['y_true_l'])
    auroc_score = metrics.print_metrics_binary(data[:, 1], data[:, 0], verbose=0)["auroc"]

    aucs = []
    for i in range(args.n_iters):
        cur_data = sk_utils.resample(data, n_samples=len(data))
        cur_auc = metrics.print_metrics_binary(cur_data[:, 1], cur_data[:, 0], verbose=0)["auroc"]
        aucs += [cur_auc]

    print "{} iterations".format(args.n_iters)
    print "ROC of AUC = {}".format(auroc_score)
    print "mean = {}".format(np.mean(aucs))
    print "median = {}".format(np.median(aucs))
    print "std = {}".format(np.std(aucs))
    print "2.5% percentile = {}".format(np.percentile(aucs, 2.5))
    print "97.5% percentile = {}".format(np.percentile(aucs, 97.5))


if __name__ == "__main__":
    main()
