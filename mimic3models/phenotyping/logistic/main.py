import numpy as np
import argparse
import time
import os
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression

from mimic3benchmark.readers import PhenotypingReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.phenotyping import utils


parser = argparse.ArgumentParser()
parser.add_argument('--period', type=str, default="all", 
                    help="first4days, first8days, last12hours, "\
                         "first25percent, first50percent, all")
parser.add_argument('--features', type=str, default="all",
                    help="all, len, all_but_len")

penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
Cs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
#penalties = ['l2']
#Cs = [1.0]

args = parser.parse_args()
print args

train_reader = PhenotypingReader(dataset_dir='../../../data/phenotyping/train/',
                    listfile='../../../data/phenotyping/train_listfile.csv')

val_reader = PhenotypingReader(dataset_dir='../../../data/phenotyping/train/',
                    listfile='../../../data/phenotyping/val_listfile.csv')

test_reader = PhenotypingReader(dataset_dir='../../../data/phenotyping/test/',
                             listfile='../../../data/phenotyping/test_listfile.csv')
                             
def read_and_extract_features(reader):
    (chunk, ts, y, header) = utils.read_chunk(reader, reader.get_number_of_examples())
    #(chunk, ts, y, header) = utils.read_chunk(reader, 200)
    X = common_utils.extract_features_from_rawdata(chunk, header, args.period, args.features)
    return (X, y)

print "==> reading data and extracting features"

(train_X, train_y) = read_and_extract_features(train_reader)
train_y = np.array(train_y)
del train_reader

(val_X, val_y) = read_and_extract_features(val_reader)
val_y = np.array(val_y)
del val_reader

(test_X, test_y) = read_and_extract_features(test_reader)
test_y = np.array(test_y)
del test_reader

print "train.shape ", train_X.shape, train_y.shape
print "val.shape", val_X.shape, val_y.shape

print "==> imputing missing values"
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0,
                  verbose=0, copy=True)
imputer.fit(train_X)
train_X = np.array(imputer.transform(train_X), dtype=np.float32)
val_X = np.array(imputer.transform(val_X), dtype=np.float32)
test_X = np.array(imputer.transform(test_X), dtype=np.float32)

print "==> normalizing data"
scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

NTASKS = 25

if not os.path.exists("activations"):
    os.mkdir("activations")

if not os.path.exists("results"):
    os.mkdir("results")

for (penalty, C) in zip(penalties, Cs):
    model_name = "%s.%s.%s.C%f" % (args.period, args.features, 
                                  penalty, C)
    
    train_activations = np.zeros(shape=train_y.shape, dtype=float)
    val_activations = np.zeros(shape=val_y.shape, dtype=float)
    test_activations = np.zeros(shape=test_y.shape, dtype=float)
    
    for task_id in range(NTASKS):
        print "==> starting task %d" % task_id
        
        file_name = ("task%d." % task_id) + model_name
        
        logreg = LogisticRegression(penalty=penalty, C=C)
        logreg.fit(train_X, train_y[:, task_id])
        
        with open(os.path.join("results", file_name + ".txt"), "w") as resfile:
    
            resfile.write("acc, prec0, prec1, rec0, rec1, auroc, auprc, minpse\n")
            
            def write_results_local(resfile, ret):
                resfile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
                    ret['acc'],
                    ret['prec0'],
                    ret['prec1'],
                    ret['rec0'],
                    ret['rec1'],
                    ret['auroc'],
                    ret['auprc'],
                    ret['minpse']))
            
            print "Scores on train set"
            train_preds = logreg.predict_proba(train_X)
            train_activations[:, task_id] = train_preds[:, 1]
            ret = metrics.print_metrics_binary(train_y[:, task_id], train_preds)
            write_results_local(resfile, ret)
                
            print "Scores on validation set"
            val_preds = logreg.predict_proba(val_X)
            val_activations[:, task_id] = val_preds[:, 1]
            ret = metrics.print_metrics_binary(val_y[:, task_id], val_preds)
            write_results_local(resfile, ret)
            
            print "Scores on test set"
            test_preds = logreg.predict_proba(test_X)
            test_activations[:, task_id] = test_preds[:, 1]
            ret = metrics.print_metrics_binary(test_y[:, task_id], test_preds)
            write_results_local(resfile, ret)
        
        
        with open(os.path.join("activations", file_name + ".txt"), "w") as actfile:
            for (x, y) in zip(test_activations[:, task_id], test_y[:, task_id]):
                actfile.write("%.6f %d\n" % (x, y))
    
    with open(os.path.join("results", model_name + ".txt"), "w") as resfile:
        header = "ave_prec_micro,ave_prec_macro,ave_prec_weighted,"
        header += "ave_recall_micro,ave_recall_macro,ave_recall_weighted,"
        header += "ave_auc_micro,ave_auc_macro,ave_auc_weighted,"
        header += ','.join(["auc_%d" % i for i in range(NTASKS)])
        resfile.write(header + "\n")
        
        def write_results(resfile, ret):
            resfile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f," % (
                ret['ave_prec_micro'], ret['ave_prec_macro'], ret['ave_prec_weighted'],
                ret['ave_recall_micro'], ret['ave_recall_macro'], ret['ave_recall_weighted'],
                ret['ave_auc_micro'], ret['ave_auc_macro'], ret['ave_auc_weighted']))
            resfile.write(",".join(["%.6f" % x for x in ret['auc_scores']]) + "\n")
        
        print "\nAverage results on train"
        ret = metrics.print_metrics_multilabel(train_y, train_activations)
        write_results(resfile, ret)
        
        print "\nAverage results on val"
        ret = metrics.print_metrics_multilabel(val_y, val_activations)
        write_results(resfile, ret)
        
        print "\nAverage results on test"
        ret = metrics.print_metrics_multilabel(test_y, test_activations)
        write_results(resfile, ret)
        
    np.savetxt(os.path.join("activations", model_name + ".csv"), test_activations, delimiter=',')
    
    print "==================== Done (penalty = %s, C = %f) ====================\n" % (penalty, C)
