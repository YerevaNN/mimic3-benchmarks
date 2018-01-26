import sklearn.utils as sk_utils
from mimic3models import metrics
import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', type=str)
    parser.add_argument('--test_listfile', type=str, default='../data/length-of-stay/test/listfile.csv')
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
    kappa_score = metrics.print_metrics_regression(data[:, 1], data[:, 0], verbose=0)["kappa"]

    kappas = []
    for i in range(args.n_iters):
        cur_data = sk_utils.resample(data, n_samples=len(data))
        ret = metrics.print_metrics_regression(cur_data[:, 0], cur_data[:, 1], verbose=0)["kappa"]
        kappas += [ret]

    print "{} iterations".format(args.n_iters)
    print "kappa score = {}".format(kappa_score)
    print "mean = {}".format(np.mean(kappas))
    print "median = {}".format(np.median(kappas))
    print "std = {}".format(np.std(kappas))
    print "2.5% percentile = {}".format(np.percentile(kappas, 2.5))
    print "97.5% percentile = {}".format(np.percentile(kappas, 97.5))
