import numpy as np
import sklearn.utils as skutils
from mimic3models import metrics
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()

    data = np.loadtxt(args.filename, delimiter=',', skiprows=1)
    n_iters = 1000
    aucs = []
    for i in range(n_iters):
        cur_data = skutils.resample(data, n_samples=len(data))
        pred = cur_data[:, 0]
        y_true = cur_data[:, 1]
        ret = metrics.print_metrics_binary(y_true, pred, verbose=0)["auroc"]
        aucs += [ret]

    print "{} iterations".format(n_iters)
    print "mean = {}".format(np.mean(aucs))
    print "median = {}".format(np.median(aucs))
    print "std = {}".format(np.std(aucs))
    print "2.5% percentile = {}".format(np.percentile(aucs, 2.5))
    print "97.5% percentile = {}".format(np.percentile(aucs, 97.5))

if __name__ == "__main__":
    main()

