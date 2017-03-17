import os
import numpy as np
import argparse
import time
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models import metrics

from mimic3models.in_hospital_mortality import utils


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--C', type=float, default=1.0,
                    help='inverse of L1 / L2 regularization')
parser.add_argument('--l1', dest='l2', action='store_false')
parser.add_argument('--l2', dest='l2', action='store_true')
parser.set_defaults(l2=True)
parser.add_argument('--period', type=str, default="all", 
                    help="first4days, first8days, last12hours, "\
                         "first25percent, first50percent, all")
parser.add_argument('--features', type=str, default="all",
                    help="all, len, all_but_len")

args = parser.parse_args()
print args

train_reader = InHospitalMortalityReader(dataset_dir='../../../data/in-hospital-mortality/train/',
                    listfile='../../../data/in-hospital-mortality/train_listfile.csv',
                    period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir='../../../data/in-hospital-mortality/train/',
                    listfile='../../../data/in-hospital-mortality/val_listfile.csv',
                    period_length=48.0)


def read_and_extract_features(reader):
    (chunk, ts, y, header) = utils.read_chunk(reader, reader.get_number_of_examples())
    #(chunk, ts, y, header) = utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(chunk, header, args.period, args.features)
    return (X, y)


print "==> reading data and extracting features"
prev_time = time.time()
(train_X, train_y) = read_and_extract_features(train_reader)
(val_X, val_y) = read_and_extract_features(val_reader)
print "train.shape ", train_X.shape
print "val.shape", val_X.shape
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
penalty = ("l2" if args.l2 else "l1")

file_name = "%s.%s.%s.C%f" % (args.period, args.features,
                              penalty, args.C)

logreg = LogisticRegression(penalty=penalty, C=args.C)
logreg.fit(train_X, train_y)

if not os.path.exists("activations"):
    os.mkdir("activations")

if not os.path.exists("results"):
    os.mkdir("results")

with open(os.path.join("results", file_name + ".txt"), "w") as resfile:
    
    resfile.write("acc, prec0, prec1, rec0, rec1, auroc, auprc, minpse\n")
    
    print "Scores on train set"
    ret = metrics.print_metrics_binary(train_y, logreg.predict_proba(train_X))
    resfile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
        ret['acc'],
        ret['prec0'],
        ret['prec1'],
        ret['rec0'],
        ret['rec1'],
        ret['auroc'],
        ret['auprc'],
        ret['minpse']))
        
    print "Scores on validation set"
    ret = metrics.print_metrics_binary(val_y, logreg.predict_proba(val_X))
    resfile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
        ret['acc'],
        ret['prec0'],
        ret['prec1'],
        ret['rec0'],
        ret['rec1'],
        ret['auroc'],
        ret['auprc'],
        ret['minpse']))


############################### TESTING #############################
# predict on test
del train_reader
del val_reader
del train_X
del val_X
del train_y
del val_y

test_reader = InHospitalMortalityReader(dataset_dir='../../../data/in-hospital-mortality/test/',
                             listfile='../../../data/in-hospital-mortality/test_listfile.csv',
                             period_length=48.0)
(test_X, test_y) = read_and_extract_features(test_reader)
test_X = np.array(imputer.transform(test_X), dtype=np.float32)
test_X = scaler.transform(test_X)

with open(os.path.join("results", file_name + ".txt"), "a") as resfile:
    print "Scores on test set"
    ret = metrics.print_metrics_binary(test_y, logreg.predict_proba(test_X))
    resfile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
        ret['acc'],
        ret['prec0'],
        ret['prec1'],
        ret['rec0'],
        ret['rec1'],
        ret['auroc'],
        ret['auprc'],
        ret['minpse']))

with open(os.path.join("activations", file_name + ".txt"), "w") as actfile:
    preds = logreg.predict_proba(test_X)[:, 1]
    for (x, y) in zip(preds, test_y):
        actfile.write("%.6f %d\n" % (x, y))
