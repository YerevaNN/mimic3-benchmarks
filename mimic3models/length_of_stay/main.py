import numpy as np
import argparse
import time
import os
import importlib
import random

from mimic3benchmark.readers import LengthOfStayReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models.length_of_stay import utils


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, required=True)
parser.add_argument('--dim', type=int, default=256,
                        help='number of hidden units')
parser.add_argument('--chunks', type=int, default=1000,
                        help='number of chunks to train')
parser.add_argument('--load_state', type=str, default="",
                        help='state file path')
parser.add_argument('--mode', type=str, default="train",
                        help='mode: train, test or info')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
parser.add_argument('--log_every', type=int, default=1,
                        help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1,
                        help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="",
                        help='optional prefix of network name')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--batch_norm', type=bool, default=False,
                        help='batch normalization')
parser.add_argument('--timestep', type=float, default=0.8,
                        help="fixed timestep used in the dataset")
parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--whole_data', dest='small_part', action='store_false')
parser.set_defaults(small_part=False)
args = parser.parse_args()
print args

train_reader = LengthOfStayReader(dataset_dir='../../data/length-of-stay/train/',
                    listfile='../../data/length-of-stay/train_listfile.csv')

val_reader = LengthOfStayReader(dataset_dir='../../data/length-of-stay/train/',
                    listfile='../../data/length-of-stay/val_listfile.csv')

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)[0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels) # choose here onlycont vs all
normalizer.load_params('los_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(args.timestep))

args_dict = dict(args._get_kwargs())

# init class
print "==> using network %s" % args.network
network_module = importlib.import_module("networks." + args.network)
network = network_module.Network(**args_dict)
time_step_suffix = ".ts%.2f" % args.timestep
network_name = args.prefix + network.say_name() + time_step_suffix
print "==> network_name:", network_name

n_trained_chunks = 0
if args.load_state != "":
    n_trained_chunks = network.load_state(args.load_state) - 1

if (args.small_part):
    args.save_every = 100000 # never save
    chunk_size = args.batch_size
else:
    chunk_size = 10000

def process_one_chunk(mode, chunk_index):
    assert (mode == "train" or mode == "test")
    
    if (mode == "train"):
        reader = train_reader
    if (mode == "test"):
        reader = val_reader
    
    (data, ts, ys, header) = utils.read_chunk(reader, chunk_size)
    data = utils.preprocess_chunk(data, ts, discretizer, normalizer)
    
    if (mode == "train"):
        network.set_datasets((data, ys), None)
    if (mode == "test"):
        network.set_datasets(None, (data, ys))
        
    network.shuffle_train_set()
        
    y_true = []
    predictions = []
    avg_loss = 0.0
    sum_loss = 0.0
    prev_time = time.time()
    n_batches = network.get_batches_per_epoch(mode)
    
    for i in range(0, n_batches):
        step_data = network.step(mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_loss_mse = step_data["loss_mse"]
        current_loss_reg = step_data["loss_reg"]
        log = step_data["log"]
        
        avg_loss += current_loss
        sum_loss += current_loss
        
        for x in answers:
            y_true.append(x)
        
        for x in prediction:
            predictions.append(x)
        
        if ((i + 1) % args.log_every == 0):
            cur_time = time.time()
            print ("  %sing: %d.%d / %d \t loss: %.3f = %.3f + %.3f \t avg_loss: %.3f \t"\
                   "%s \t time: %.2fs" % (mode, chunk_index, i * args.batch_size,
                        n_batches * args.batch_size, 
                        current_loss, current_loss_mse, current_loss_reg,
                        avg_loss / args.log_every, log, cur_time - prev_time))
            avg_loss = 0
            prev_time = cur_time
        
        if np.isnan(current_loss):
            raise Exception ("current loss IS NaN. This should never happen :)") 

    sum_loss /= n_batches
    print "\n  %s loss = %.5f" % (mode, sum_loss)
    
    if args.network in ['lstm', 'lstm_log']:
        metrics.print_metrics_regression(y_true, predictions)
    if args.network == 'lstm_cf_log':
        metrics.print_metrics_log_bins(y_true, predictions)
    if args.network == 'lstm_cf_custom':
        metrics.print_metrics_custom_bins(y_true, predictions)
    
    return sum_loss


if args.mode == 'train':
    
    print "==> training"  	
    for chunk_index in range(n_trained_chunks, n_trained_chunks + args.chunks):
        start_time = time.time()
        
        process_one_chunk("train", chunk_index)
        cnt_trained = chunk_index - n_trained_chunks + 1

        if (cnt_trained % 5 == 0):
            val_loss = process_one_chunk("test", chunk_index)
            if ((cnt_trained / 5) % args.save_every == 0):
                state_name = 'states/%s.chunk%d.test%.8f.state' % (network_name,
                                        chunk_index, val_loss)
                               
                print "==> saving ... %s" % state_name
                network.save_params(state_name, chunk_index)
        
        print "chunk %d took %.3fs" % (chunk_index, float(time.time()) - start_time)
        
        chunks_per_epoch = train_reader.get_number_of_examples() // chunk_size
        if (cnt_trained % chunks_per_epoch == 0):
            train_reader.random_shuffle()
            val_reader.random_shuffle()
            
elif args.mode == 'test':
    # ensure that the code uses test_reader
    del train_reader
    del val_reader 
    
    test_reader = LengthOfStayReader(dataset_dir='../../data/length-of-stay/test/',
            listfile='../../data/length-of-stay/test_listfile.csv')
    
    n_batches = test_reader.get_number_of_examples() // args.batch_size
    y_true = []
    predictions = []
    avg_loss = 0.0
    sum_loss = 0.0
    prev_time = time.time()
    
    n_batches = 1000 # TODO: remove this, to test on full data
    
    for i in range(n_batches):
        (data, ts, ys, header) = utils.read_chunk(test_reader, args.batch_size)
        data = utils.preprocess_chunk(data, ts, discretizer, normalizer)
        ret = network.predict((data, ys))
        prediction = ret[0]
        current_loss = ret[1]
        
        avg_loss += current_loss
        sum_loss += current_loss
        
        for x in ys:
            y_true.append(x)
        
        for x in prediction:
            predictions.append(x)
        
        if ((i + 1) % args.log_every == 0):
            cur_time = time.time()
            print ("  testing: %d / %d \t loss: %.3f \t avg_loss: %.3f \t"\
                   " time: %.2fs" % ((i+1) * args.batch_size,
                        n_batches * args.batch_size, current_loss,
                        avg_loss / args.log_every, cur_time - prev_time))
            avg_loss = 0
            prev_time = cur_time
        
        if np.isnan(current_loss):
            raise Exception ("current loss IS NaN. This should never happen :)") 

    sum_loss /= n_batches
    print "\n  test loss = %.5f" % sum_loss
    
    if args.network in ['lstm', 'lstm_log']:
        metrics.print_metrics_regression(y_true, predictions)
    if args.network == 'lstm_cf_log':
        metrics.print_metrics_log_bins(y_true, predictions)
    if args.network == 'lstm_cf_custom':
        metrics.print_metrics_custom_bins(y_true, predictions)
    
    with open("activations.txt", "w") as fout:
        fout.write("prediction, y_true")
        for (x, y) in zip(predictions, y_true):
            fout.write("%.6f, %.6f\n" % (x, y))
    
else:
    raise Exception("unknown mode")