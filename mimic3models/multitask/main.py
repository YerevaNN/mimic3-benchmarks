import numpy as np
import argparse
import time
import os
import importlib
import random
import copy

from mimic3benchmark.readers import MultitaskReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models.multitask import utils


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, required=True)
parser.add_argument('--dim', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
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
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                    help="shuffles the training set")
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true',
                    help='batch normalization')
parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--whole_data', dest='small_part', action='store_false')
parser.add_argument('--timestep', type=str, default="1.0",
                    help="fixed timestep used in the dataset")
parser.add_argument('--ihm_C', type=float, default=1.0)
parser.add_argument('--los_C', type=float, default=1.0)
parser.add_argument('--ph_C', type=float, default=1.0)
parser.add_argument('--decomp_C', type=float, default=1.0)
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--partition', type=str, default="custom", help="log or custom")
parser.add_argument('--nbins', type=int, default=10)

parser.set_defaults(shuffle=True)
parser.set_defaults(batch_norm=True)
parser.set_defaults(small_part=False)
args = parser.parse_args()
print args

train_reader = MultitaskReader(dataset_dir='../../data/multitask/train/',
                                  listfile='../../data/multitask/train_listfile.csv')

val_reader = MultitaskReader(dataset_dir='../../data/multitask/train/',
                                 listfile='../../data/multitask/val_listfile.csv')

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)[0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels) # choose here onlycont vs all
normalizer.load_params('mult_ts%s.input_str:%s.start_time:zero.normalizer' % (args.timestep, args.imputation))

train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
test_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

args_dict = dict(args._get_kwargs())
args_dict['train_raw'] = train_raw
args_dict['test_raw'] = test_raw

# init class
print "==> using network %s" % args.network
network_module = importlib.import_module("networks." + args.network)
network = network_module.Network(**args_dict)
time_step_suffix = ".ts%s" % args.timestep
network_name = args.prefix + network.say_name() + time_step_suffix + "." + args.imputation
print "==> network_name:", network_name


start_epoch = -1
if args.load_state != "":
    start_epoch = network.load_state(args.load_state)

def do_epoch(mode, epoch):
    # mode is 'train' or 'test'

    ihm_predictions = []
    ihm_answers = []
    
    los_predictions = []
    los_answers = []
    
    ph_predictions = []
    ph_answers = []
    
    decomp_predictions = []
    decomp_answers = []
    
    avg_loss = 0.0
    sum_loss = 0.0
    prev_time = time.time()
    
    batches_per_epoch = network.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = network.step(mode)
        
        ihm_pred = step_data["ihm_prediction"]
        los_pred = step_data["los_prediction"]
        ph_pred = step_data["ph_prediction"]
        decomp_pred = step_data["decomp_prediction"]
        
        current_loss = step_data["loss"]
        ihm_loss = step_data["ihm_loss"]
        los_loss = step_data["los_loss"]
        ph_loss = step_data["ph_loss"]
        decomp_loss = step_data["decomp_loss"]
        reg_loss = step_data["reg_loss"]
        
        data = step_data["data"]
        
        ihm_data = data[1]
        ihm_mask = [x[1] for x in ihm_data]
        ihm_label = [x[2] for x in ihm_data]
        
        los_data = data[2]
        los_mask = [x[0] for x in los_data]
        los_label = [x[1] for x in los_data]
        
        ph_data = data[3]
        ph_label = ph_data
        
        decomp_data = data[4]
        decomp_mask = [x[0] for x in decomp_data]
        decomp_label = [x[1] for x in decomp_data]
        
        avg_loss += current_loss
        sum_loss += current_loss
        
        for (x, mask, y) in zip(ihm_pred, ihm_mask, ihm_label):
            if (mask == 1):
                ihm_predictions.append(x)
                ihm_answers.append(y)
        
        for (sx, smask, sy) in zip(los_pred, los_mask, los_label):
            for (x, mask, y) in zip(sx, smask, sy):
                if (mask == 1):
                    los_predictions.append(x)
                    los_answers.append(y)

        for (x, y) in zip(ph_pred, ph_label):
            ph_predictions.append(x)
            ph_answers.append(y)
            
        for (sx, smask, sy) in zip(decomp_pred, decomp_mask, decomp_label):
            for (x, mask, y) in zip(sx, smask, sy):
                if (mask == 1):
                    decomp_predictions.append(x)
                    decomp_answers.append(y)
        
        if ((i + 1) % args.log_every == 0):
            cur_time = time.time()
            print "  {}ing {}.{} / {}  loss: {:8.4f} = {:1.2f} + {:8.2f} + {:1.2f} + "\
                  "{:1.2f} + {:.2f} avg_loss: {:6.4f}  time: {:6.4f}".format(
                        mode, epoch, i * args.batch_size,
                        batches_per_epoch * args.batch_size,
                        float(current_loss),
                        float(ihm_loss), float(los_loss), float(ph_loss),
                        float(decomp_loss), float(reg_loss),
                        float(avg_loss / args.log_every),
                        float(cur_time - prev_time))
            avg_loss = 0
            prev_time = cur_time
        
        if np.isnan(current_loss):
            print "loss: {:6.4f} = {:1.2f} + {:8.2f} + {:1.2f} + {:1.2f} + {:.2f}".format(
                    float(current_loss),
                    float(ihm_loss), float(los_loss), float(ph_loss),
                    float(decomp_loss), float(reg_loss))
            raise Exception ("current loss IS NaN. This should never happen :)") 

    sum_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, sum_loss)
    
    eps = 1e-13
    if args.ihm_C > eps:
        print "\n ================= 48h mortality ================"
        metrics.print_metrics_binary(ihm_answers, ihm_predictions)
    
    if args.los_C > eps:
        print "\n ================ length of stay ================"
        if args.partition == 'log':
            metrics.print_metrics_log_bins(los_answers, los_predictions)
        else:
            metrics.print_metrics_custom_bins(los_answers, los_predictions)
    
    if args.ph_C > eps:    
        print "\n =================== phenotype =================="
        metrics.print_metrics_multilabel(ph_answers, ph_predictions)
    
    if args.decomp_C > eps:
        print "\n ================ decompensation ================"
        metrics.print_metrics_binary(decomp_answers, decomp_predictions)
    
    return sum_loss


if args.mode == 'train':
    print "==> training"   	
    for epoch in range(start_epoch + 1, start_epoch + 1 + args.epochs):
        start_time = time.time()
        
        if args.shuffle:
            network.shuffle_train_set()
        
        do_epoch('train', epoch)
        epoch_loss = do_epoch('test', epoch)

        state_name = 'states/%s.epoch%d.test%.8f.state' % (network_name, epoch,
                                                           epoch_loss)
        if ((epoch + 1) % args.save_every == 0):    
            print "==> saving ... %s" % state_name
            network.save_params(state_name, epoch)
        
        print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)

elif args.mode == 'test':
    # ensure that the code uses test_reader
    del train_reader
    del val_reader 
    
    test_reader = MultitaskReader(dataset_dir='../../data/multitask/test/',
                                      listfile='../../data/multitask/test_listfile.csv')
    data_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part)
    
    data_raw_copy = copy.deepcopy(data_raw) # TODO: delete this 
    
    ihm_predictions = []
    ihm_answers = []
    
    los_predictions = []
    los_answers = []
    
    ph_predictions = []
    ph_answers = []
    
    decomp_predictions = []
    decomp_answers = []
    
    avg_loss = 0.0
    sum_loss = 0.0
    prev_time = time.time()
    
    batches_per_epoch = len(data_raw[0]) // args.batch_size
    
    for i in range(0, batches_per_epoch):
        start = i * args.batch_size
        end = start + args.batch_size
        
        data = (data_raw[0][start:end],
                data_raw[1][start:end],
                data_raw[2][start:end],
                data_raw[3][start:end],
                data_raw[4][start:end])
        
        ret = network.predict(data)
        
        data = (data_raw_copy[0][start:end],
                data_raw_copy[1][start:end],
                data_raw_copy[2][start:end],
                data_raw_copy[3][start:end],
                data_raw_copy[4][start:end]) # TODO: delete this
        
        ihm_pred = ret["ihm_prediction"]
        los_pred = ret["los_prediction"]
        ph_pred = ret["ph_prediction"]
        decomp_pred = ret["decomp_prediction"]
        
        current_loss = ret["loss"]
        ihm_loss = ret["ihm_loss"]
        los_loss = ret["los_loss"]
        ph_loss = ret["ph_loss"]
        decomp_loss = ret["decomp_loss"]
        reg_loss = ret["reg_loss"]
        
        ihm_data = data[1]
        ihm_mask = [x[1] for x in ihm_data]
        ihm_label = [x[2] for x in ihm_data]
        
        los_data = data[2]
        los_mask = [x[0] for x in los_data]
        los_label = [x[1] for x in los_data]
        
        ph_data = data[3]
        ph_label = ph_data
        
        decomp_data = data[4]
        decomp_mask = [x[0] for x in decomp_data]
        decomp_label = [x[1] for x in decomp_data]
        
        avg_loss += current_loss
        sum_loss += current_loss
        
        for (x, mask, y) in zip(ihm_pred, ihm_mask, ihm_label):
            if (mask == 1):
                ihm_predictions.append(x)
                ihm_answers.append(y)
        
        for (sx, smask, sy) in zip(los_pred, los_mask, los_label):
            for (x, mask, y) in zip(sx, smask, sy):
                if (mask == 1):
                    los_predictions.append(x)
                    los_answers.append(y)

        for (x, y) in zip(ph_pred, ph_label):
            ph_predictions.append(x)
            ph_answers.append(y)
            
        for (sx, smask, sy) in zip(decomp_pred, decomp_mask, decomp_label):
            for (x, mask, y) in zip(sx, smask, sy):
                if (mask == 1):
                    decomp_predictions.append(x)
                    decomp_answers.append(y)
        
        if ((i + 1) % args.log_every == 0):
            cur_time = time.time()
            print "  {}ing {} / {}  loss: {:8.4f} = {:1.2f} + {:8.2f} + {:1.2f} + "\
                  "{:1.2f} + {:.2f} avg_loss: {:6.4f}  time: {:6.4f}".format(
                        args.mode, i * args.batch_size,
                        batches_per_epoch * args.batch_size,
                        float(current_loss),
                        float(ihm_loss), float(los_loss), float(ph_loss),
                        float(decomp_loss), float(reg_loss),
                        float(avg_loss / args.log_every),
                        float(cur_time - prev_time))
            avg_loss = 0
            prev_time = cur_time
        
        if np.isnan(current_loss):
            print "loss: {:6.4f} = {:1.2f} + {:8.2f} + {:1.2f} + {:1.2f} + {:.2f}".format(
                    float(current_loss),
                    float(ihm_loss), float(los_loss), float(ph_loss),
                    float(decomp_loss), float(reg_loss))
            raise Exception ("current loss IS NaN. This should never happen :)") 

    sum_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (args.mode, sum_loss)
    
    eps = 1e-13
    if args.ihm_C > eps:
        print "\n ================= 48h mortality ================"
        metrics.print_metrics_binary(ihm_answers, ihm_predictions)
    
    if args.los_C > eps:
        print "\n ================ length of stay ================"
        if args.partition == 'log':
            metrics.print_metrics_log_bins(los_answers, los_predictions)
        else:
            metrics.print_metrics_custom_bins(los_answers, los_predictions)

    if args.ph_C > eps:
        print "\n =================== phenotype =================="
        metrics.print_metrics_multilabel(ph_answers, ph_predictions)
    
    if args.decomp_C > eps:
        print "\n ================ decompensation ================"
        metrics.print_metrics_binary(decomp_answers, decomp_predictions)
    
    with open("los_activations.txt", "w") as fout:
        fout.write("prediction, y_true")
        for (x, y) in zip(los_predictions, los_answers):
            fout.write("%.6f, %.6f\n" % (x, y))
    
else:
    raise Exception("unknown mode")