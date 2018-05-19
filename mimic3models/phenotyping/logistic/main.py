from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from mimic3benchmark.readers import PhenotypingReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.phenotyping.utils import save_results

import numpy as np
import argparse
import os
import json


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'], ret['t'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--grid-search', dest='grid_search', action='store_true')
    parser.add_argument('--no-grid-search', dest='grid_search', action='store_false')
    parser.set_defaults(grid_search=False)
    parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/phenotyping/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    if args.grid_search:
        penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
        coefs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
    else:
        penalties = ['l1']
        coefs = [0.1]

    train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                     listfile=os.path.join(args.data, 'train_listfile.csv'))

    val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                   listfile=os.path.join(args.data, 'val_listfile.csv'))

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

    print('Reading data and extracting features ...')

    (train_X, train_y, train_names, train_ts) = read_and_extract_features(train_reader, args.period, args.features)
    train_y = np.array(train_y)

    (val_X, val_y, val_names, val_ts) = read_and_extract_features(val_reader, args.period, args.features)
    val_y = np.array(val_y)

    (test_X, test_y, test_names, test_ts) = read_and_extract_features(test_reader, args.period, args.features)
    test_y = np.array(test_y)

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

    n_tasks = 25
    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    for (penalty, C) in zip(penalties, coefs):
        model_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, C)

        train_activations = np.zeros(shape=train_y.shape, dtype=float)
        val_activations = np.zeros(shape=val_y.shape, dtype=float)
        test_activations = np.zeros(shape=test_y.shape, dtype=float)

        for task_id in range(n_tasks):
            print('Starting task {}'.format(task_id))

            logreg = LogisticRegression(penalty=penalty, C=C, random_state=42)
            logreg.fit(train_X, train_y[:, task_id])

            train_preds = logreg.predict_proba(train_X)
            train_activations[:, task_id] = train_preds[:, 1]

            val_preds = logreg.predict_proba(val_X)
            val_activations[:, task_id] = val_preds[:, 1]

            test_preds = logreg.predict_proba(test_X)
            test_activations[:, task_id] = test_preds[:, 1]

        with open(os.path.join(result_dir, 'train_{}.json'.format(model_name)), 'w') as f:
            ret = metrics.print_metrics_multilabel(train_y, train_activations)
            ret = {k: float(v) for k, v in ret.items() if k != 'auc_scores'}
            json.dump(ret, f)

        with open(os.path.join(result_dir, 'val_{}.json'.format(model_name)), 'w') as f:
            ret = metrics.print_metrics_multilabel(val_y, val_activations)
            ret = {k: float(v) for k, v in ret.items() if k != 'auc_scores'}
            json.dump(ret, f)

        with open(os.path.join(result_dir, 'test_{}.json'.format(model_name)), 'w') as f:
            ret = metrics.print_metrics_multilabel(test_y, test_activations)
            ret = {k: float(v) for k, v in ret.items() if k != 'auc_scores'}
            json.dump(ret, f)

        save_results(test_names, test_ts, test_activations, test_y,
                     os.path.join(args.output_dir, 'predictions', model_name + '.csv'))


if __name__ == '__main__':
    main()
