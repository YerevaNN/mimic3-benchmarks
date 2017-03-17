import numpy as np
import argparse
import time
import os
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression

from mimic3benchmark.readers import LengthOfStayReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.length_of_stay import utils


parser = argparse.ArgumentParser()
parser.add_argument('--period', type=str, default="all", 
                    help="first4days, first8days, last12hours, "\
                         "first25percent, first50percent, all")
parser.add_argument('--features', type=str, default="all",
                    help="all, len, all_but_len")

#penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
#Cs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
penalties = ['l2']
Cs = [0.00001]

args = parser.parse_args()
print args


nbins = 10
NTASKS = 10

def one_hot(index):
    x = np.zeros((nbins,), dtype=np.int32)
    x[index] = 1
    return x


train_reader = LengthOfStayReader(dataset_dir='../../../data/length-of-stay/train/',
                    listfile='../../../data/length-of-stay/train_listfile.csv')

val_reader = LengthOfStayReader(dataset_dir='../../../data/length-of-stay/train/',
                    listfile='../../../data/length-of-stay/val_listfile.csv')


test_reader = LengthOfStayReader(dataset_dir='../../../data/length-of-stay/test/',
                             listfile='../../../data/length-of-stay/test_listfile.csv')


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
    bins = np.array([one_hot(metrics.get_bin_custom(x, nbins)) for x in ys])
    return (Xs, bins, ys)


print "==> reading data and extracting features"
chunk_size = 100000 # TODO: bigger chunk_size

prev_time = time.time()
(train_X, train_y, train_actual) = read_and_extract_features(train_reader, chunk_size)
del train_reader

(val_X, val_y, val_actual) = read_and_extract_features(val_reader, chunk_size)
del val_reader

(test_X, test_y, test_actual) = read_and_extract_features(test_reader,
                                                          test_reader.get_number_of_examples())
del test_reader

print "==> elapsed time = %.3f" % (time.time() - prev_time)

print "train.shape ", train_X.shape, train_y.shape
print "val.shape", val_X.shape, val_y.shape
print "test.shape", test_X.shape, test_y.shape

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
        
        with open(os.path.join("cf_results", file_name + ".txt"), "w") as resfile:
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
            
            print "Task specific scores on train set"
            train_preds = logreg.predict_proba(train_X)
            train_activations[:, task_id] = train_preds[:, 1]
            ret = metrics.print_metrics_binary(train_y[:, task_id], train_preds)
            write_results_local(resfile, ret)
                
            print "Task specific scores on validation set"
            val_preds = logreg.predict_proba(val_X)
            val_activations[:, task_id] = val_preds[:, 1]
            ret = metrics.print_metrics_binary(val_y[:, task_id], val_preds)
            write_results_local(resfile, ret)
            
            print "Task specific scores on test set"
            test_preds = logreg.predict_proba(test_X)
            test_activations[:, task_id] = test_preds[:, 1]
            ret = metrics.print_metrics_binary(test_y[:, task_id], test_preds)
            write_results_local(resfile, ret)
    
    
    with open(os.path.join("cf_results", "classification_" + model_name + ".txt"), "w") as resfile:
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
    
    
    train_predictions = np.array([metrics.get_estimate_custom(x, nbins) for x in train_activations])
    val_predictions = np.array([metrics.get_estimate_custom(x, nbins) for x in val_activations])
    test_predictions = np.array([metrics.get_estimate_custom(x, nbins) for x in test_activations])
    
    with open(os.path.join("cf_activations", model_name + ".txt"), "w") as actfile:
        for (x, y) in zip(test_predictions, test_actual):
            actfile.write("%.6f %.6f\n" % (x, y))
    
    with open(os.path.join("cf_results", model_name + ".txt"), "w") as resfile:
        resfile.write("mad, mse, mape, kappa\n")
        
        print "Scores on train set"
        ret = metrics.print_metrics_custom_bins(train_actual, train_predictions)
        resfile.write("%.6f,%.6f,%.6f,%.6f\n" % (
            ret['mad'],
            ret['mse'],
            ret['mape'],
            ret['kappa']))
    
        print "Scores on val set"
        ret = metrics.print_metrics_custom_bins(val_actual, val_predictions)
        resfile.write("%.6f,%.6f,%.6f,%.6f\n" % (
            ret['mad'],
            ret['mse'],
            ret['mape'],
            ret['kappa']))
    
        print "Scores on test set"
        ret = metrics.print_metrics_custom_bins(test_actual, test_predictions)
        resfile.write("%.6f,%.6f,%.6f,%.6f\n" % (
            ret['mad'],
            ret['mse'],
            ret['mape'],
            ret['kappa']))
    
    print "==================== Done (penalty = %s, C = %f) ====================\n" % (penalty, C)
