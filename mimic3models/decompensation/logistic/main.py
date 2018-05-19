from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from mimic3benchmark.readers import DecompensationReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.decompensation.utils import save_results

import os
import numpy as np
import argparse
import json


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
    return (Xs, ys, names, ts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--grid-search', dest='grid_search', action='store_true')
    parser.add_argument('--no-grid-search', dest='grid_search', action='store_false')
    parser.set_defaults(grid_search=False)
    parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/decompensation/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    if args.grid_search:
        penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
        coefs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
    else:
        penalties = ['l2']
        coefs = [0.001]

    train_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'train_listfile.csv'))

    val_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'val_listfile.csv'))

    test_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'test'),
                                       listfile=os.path.join(args.data, 'test_listfile.csv'))

    print('Reading data and extracting features ...')
    n_train = min(100000, train_reader.get_number_of_examples())
    n_val = min(100000, val_reader.get_number_of_examples())

    (train_X, train_y, train_names, train_ts) = read_and_extract_features(
        train_reader, n_train, args.period, args.features)

    (val_X, val_y, val_names, val_ts) = read_and_extract_features(
        val_reader, n_val, args.period, args.features)

    (test_X, test_y, test_names, test_ts) = read_and_extract_features(
        test_reader, test_reader.get_number_of_examples(), args.period, args.features)

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

    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    for (penalty, C) in zip(penalties, coefs):
        file_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, C)

        logreg = LogisticRegression(penalty=penalty, C=C, random_state=42)
        logreg.fit(train_X, train_y)

        with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), "w") as res_file:
            ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, res_file)

        with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
            ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, res_file)

        prediction = logreg.predict_proba(test_X)[:, 1]

        with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
            ret = print_metrics_binary(test_y, prediction)
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, res_file)

        save_results(test_names, test_ts, prediction, test_y,
                     os.path.join(args.output_dir, 'predictions', file_name + '.csv'))


if __name__ == '__main__':
    main()
