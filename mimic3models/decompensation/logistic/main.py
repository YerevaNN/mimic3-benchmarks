import os
import numpy as np
import argparse
import time
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression

from mimic3benchmark.readers import DecompensationReader
from mimic3models import common_utils
from mimic3models import metrics

from mimic3models.decompensation import utils


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--period', type=str, default="all", 
                    help="first4days, first8days, last12hours, "\
                         "first25percent, first50percent, all")
parser.add_argument('--features', type=str, default="all",
                    help="all, len, all_but_len")

#penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
#Cs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
penalties = ['l2']
Cs = [0.001]

args = parser.parse_args()
print args

train_reader = DecompensationReader(dataset_dir='../../../data/decompensation/train/',
                    listfile='../../../data/decompensation/train_listfile.csv')

val_reader = DecompensationReader(dataset_dir='../../../data/decompensation/train/',
                    listfile='../../../data/decompensation/val_listfile.csv')

test_reader = DecompensationReader(dataset_dir='../../../data/decompensation/test/',
                             listfile='../../../data/decompensation/test_listfile.csv')

def read_and_extract_features(reader, count):
    read_chunk_size = 1000
    #assert (count % read_chunk_size == 0)
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

(train_X, train_y) = read_and_extract_features(train_reader, chunk_size)
del train_reader

(val_X, val_y) = read_and_extract_features(val_reader, chunk_size)
del val_reader

(test_X, test_y) = read_and_extract_features(test_reader,
                                             test_reader.get_number_of_examples())
del test_reader


print "==> imputing missing values"
# imput missing values
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0,
                  verbose=0, copy=True)
imputer.fit(train_X)
train_X = np.array(imputer.transform(train_X), dtype=np.float32)
val_X = np.array(imputer.transform(val_X), dtype=np.float32)
test_X = np.array(imputer.transform(test_X), dtype=np.float32)

print "==> normalizing data"
# shift and scale to have zero mean and unit variance
scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

if not os.path.exists("activations"):
    os.mkdir("activations")

if not os.path.exists("results"):
    os.mkdir("results")

for (penalty, C) in zip(penalties, Cs):
    file_name = "%s.%s.%s.C%f" % (args.period, args.features, 
                                  penalty, C)
    
    logreg = LogisticRegression(penalty=penalty, C=C)
    logreg.fit(train_X, train_y)

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

    print "==================== Done (penalty = %s, C = %f) ====================\n" % (penalty, C)