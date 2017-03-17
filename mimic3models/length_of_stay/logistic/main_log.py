import os
import numpy as np
import argparse
import time
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LinearRegression

from mimic3benchmark.readers import LengthOfStayReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.length_of_stay import utils


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--period', type=str, default="all", 
                    help="first4days, first8days, last12hours, "\
                         "first25percent, first50percent, all")
parser.add_argument('--features', type=str, default="all",
                    help="all, len, all_but_len")

args = parser.parse_args()
print args

train_reader = LengthOfStayReader(dataset_dir='../../../data/length-of-stay/train/',
                    listfile='../../../data/length-of-stay/train_listfile.csv')

val_reader = LengthOfStayReader(dataset_dir='../../../data/length-of-stay/train/',
                    listfile='../../../data/length-of-stay/val_listfile.csv')


def read_and_extract_features(reader, count):
    read_chunk_size = 1000
    assert (count % read_chunk_size == 0)
    Xs = []
    ys = []
    for i in range(count // read_chunk_size):
        (chunk, ts, y, header) = utils.read_chunk(reader, read_chunk_size)
        X = common_utils.extract_features_from_rawdata(chunk, header, args.period, args.features)
        Xs.append(X)
        ys += y
    Xs = np.concatenate(Xs, axis=0)
    return (Xs, ys)

print "==> reading data and extracting features"
chunk_size = 100000 # TODO: bigger chunk_size
prev_time = time.time()

(train_X, train_y) = read_and_extract_features(train_reader, chunk_size)
train_y_log = np.log(np.array(train_y) + 1)

(val_X, val_y) = read_and_extract_features(val_reader, chunk_size)
val_y_log = np.log(np.array(val_y) + 1)

print np.mean(val_y_log), np.mean(train_y_log)

print "==> elapsed time = %.3f" % (time.time() - prev_time)


print "==> imputing missing values"
# imput missing values
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0,
                  verbose=0, copy=True)
imputer.fit(train_X)
train_X = np.array(imputer.transform(train_X), dtype=np.float32)
val_X = np.array(imputer.transform(val_X), dtype=np.float32)

print "==> normalizing data"
# shift and scale to have zero mean and unit variance
scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)

file_name = "%s.%s" % (args.period, args.features)

linreg = LinearRegression()
linreg.fit(train_X, train_y_log)

if not os.path.exists("activations"):
    os.mkdir("activations")

if not os.path.exists("results"):
    os.mkdir("results")

with open(os.path.join("results", "log_" + file_name + ".txt"), "w") as resfile:
    
    resfile.write("mad, mse, mape, kappa\n")
    
    print "Scores on train set"
    pred = linreg.predict(train_X)
    pred[pred > 8] = 8
    ret = metrics.print_metrics_regression(train_y, np.exp(pred) - 1)
    resfile.write("%.6f,%.6f,%.6f,%.6f\n" % (
        ret['mad'],
        ret['mse'],
        ret['mape'],
        ret['kappa']))
    
    print "Scores on validation set"
    pred = linreg.predict(val_X)
    pred[pred > 8] = 8
    ret = metrics.print_metrics_regression(val_y, np.exp(pred) - 1)
    resfile.write("%.6f,%.6f,%.6f,%.6f\n" % (
        ret['mad'],
        ret['mse'],
        ret['mape'],
        ret['kappa']))

############################### TESTING #############################
# predict on test
del train_reader
del val_reader
del train_X
del val_X
del train_y
del val_y

test_reader = LengthOfStayReader(dataset_dir='../../../data/length-of-stay/test/',
                             listfile='../../../data/length-of-stay/test_listfile.csv')
(test_X, test_y) = read_and_extract_features(test_reader, chunk_size)
test_y_log = np.log(np.array(test_y) + 1)
test_X = np.array(imputer.transform(test_X), dtype=np.float32)
test_X = scaler.transform(test_X)

with open(os.path.join("results", "log_" + file_name + ".txt"), "a") as resfile:
    print "Scores on test set"
    pred = linreg.predict(test_X)
    pred[pred > 8] = 8
    ret = metrics.print_metrics_regression(test_y, np.exp(pred) - 1)
    resfile.write("%.6f,%.6f,%.6f,%.6f\n" % (
        ret['mad'],
        ret['mse'],
        ret['mape'],
        ret['kappa']))

with open(os.path.join("activations", "log_" + file_name + ".txt"), "w") as actfile:
    preds = linreg.predict(test_X)
    for (x, y) in zip(preds, test_y):
        actfile.write("%.6f %.6f\n" % (x, y))
