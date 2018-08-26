from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from mimic3benchmark.readers import LengthOfStayReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.length_of_stay.utils import save_results

import numpy as np
import argparse
import os
import json

n_bins = 10


def one_hot(index):
    x = np.zeros((n_bins,), dtype=np.int32)
    x[index] = 1
    return x


def read_and_extract_features(reader, count, period, features):
    read_chunk_size = 1000
    Xs = []
    ys = []
    names = []
    ts = []
    for i in range(0, count, read_chunk_size):
        j = min(count, i + read_chunk_size)
        ret = common_utils.read_chunk(reader, j - i)
        X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
        Xs.append(X)
        ys += ret['y']
        names += ret['name']
        ts += ret['t']
    Xs = np.concatenate(Xs, axis=0)
    bins = np.array([one_hot(metrics.get_bin_custom(x, n_bins)) for x in ys])
    return (Xs, bins, ys, names, ts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--grid-search', dest='grid_search', action='store_true')
    parser.add_argument('--no-grid-search', dest='grid_search', action='store_false')
    parser.set_defaults(grid_search=False)
    parser.add_argument('--data', type=str, help='Path to the data of length-of-stay task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/length-of-stay/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    if args.grid_search:
        penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
        coefs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
    else:
        penalties = ['l2']
        coefs = [0.00001]

    train_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'train_listfile.csv'))

    val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                    listfile=os.path.join(args.data, 'val_listfile.csv'))

    test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'test'),
                                     listfile=os.path.join(args.data, 'test_listfile.csv'))

    print('Reading data and extracting features ...')
    n_train = min(100000, train_reader.get_number_of_examples())
    n_val = min(100000, val_reader.get_number_of_examples())

    (train_X, train_y, train_actual, train_names, train_ts) = read_and_extract_features(
        train_reader, n_train, args.period, args.features)

    (val_X, val_y, val_actual, val_names, val_ts) = read_and_extract_features(
        val_reader, n_val, args.period, args.features)

    (test_X, test_y, test_actual, test_names, test_ts) = read_and_extract_features(
        test_reader, test_reader.get_number_of_examples(), args.period, args.features)

    print("train set shape:  {}".format(train_X.shape))
    print("validation set shape: {}".format(val_X.shape))
    print("test set shape: {}".format(test_X.shape))

    print('Imputing missing values ...')
    imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    result_dir = os.path.join(args.output_dir, 'cf_results')
    common_utils.create_directory(result_dir)

    for (penalty, C) in zip(penalties, coefs):
        model_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, C)

        train_activations = np.zeros(shape=train_y.shape, dtype=float)
        val_activations = np.zeros(shape=val_y.shape, dtype=float)
        test_activations = np.zeros(shape=test_y.shape, dtype=float)

        for task_id in range(n_bins):
            logreg = LogisticRegression(penalty=penalty, C=C, random_state=42)
            logreg.fit(train_X, train_y[:, task_id])

            train_preds = logreg.predict_proba(train_X)
            train_activations[:, task_id] = train_preds[:, 1]

            val_preds = logreg.predict_proba(val_X)
            val_activations[:, task_id] = val_preds[:, 1]

            test_preds = logreg.predict_proba(test_X)
            test_activations[:, task_id] = test_preds[:, 1]

        train_predictions = np.array([metrics.get_estimate_custom(x, n_bins) for x in train_activations])
        val_predictions = np.array([metrics.get_estimate_custom(x, n_bins) for x in val_activations])
        test_predictions = np.array([metrics.get_estimate_custom(x, n_bins) for x in test_activations])

        with open(os.path.join(result_dir, 'train_{}.json'.format(model_name)), 'w') as f:
            ret = metrics.print_metrics_custom_bins(train_actual, train_predictions)
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, f)

        with open(os.path.join(result_dir, 'val_{}.json'.format(model_name)), 'w') as f:
            ret = metrics.print_metrics_custom_bins(val_actual, val_predictions)
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, f)

        with open(os.path.join(result_dir, 'test_{}.json'.format(model_name)), 'w') as f:
            ret = metrics.print_metrics_custom_bins(test_actual, test_predictions)
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, f)

        save_results(test_names, test_ts, test_predictions, test_actual,
                     os.path.join(args.output_dir, 'cf_predictions', model_name + '.csv'))


if __name__ == '__main__':
    main()
