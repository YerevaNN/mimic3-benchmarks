import numpy as np
import argparse
import time
import os
import importlib

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics


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
parser.add_argument('--timestep', type=str, default="0.8",
                    help="fixed timestep used in the dataset")
parser.add_argument('--imputation', type=str, default='previous')

parser.set_defaults(shuffle=True)
parser.set_defaults(batch_norm=True)
parser.set_defaults(small_part=False)
args = parser.parse_args()
print args

train_reader = InHospitalMortalityReader(dataset_dir='../../data/in-hospital-mortality/train/',
                                        listfile='../../data/in-hospital-mortality/train_listfile.csv',
                                        period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir='../../data/in-hospital-mortality/train/',
                                      listfile='../../data/in-hospital-mortality/val_listfile.csv',
                                      period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)[0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels) # choose here onlycont vs all
normalizer.load_params('ihm_ts%s.input_str:%s.start_time:zero.normalizer' % (args.timestep, args.imputation))
#normalizer=None

train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
test_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

args_dict = dict(args._get_kwargs())
args_dict['train_raw'] = train_raw
args_dict['test_raw'] = test_raw
args_dict['header'] = discretizer_header

# init class
print "==> using network %s" % args.network
network_module = importlib.import_module("networks." + args.network)
network = network_module.Network(**args_dict)
time_step_suffix = ".ts%s" % args.timestep
network_name = args.prefix + network.say_name() + time_step_suffix
print "==> network_name:", network_name

start_epoch = -1
if args.load_state != "":
    start_epoch = network.load_state(args.load_state)

def do_epoch(mode, epoch):
    # mode is 'train' or 'test'
    y_true = []
    predictions = []
    
    avg_loss = 0.0
    sum_loss = 0.0
    prev_time = time.time()
    
    batches_per_epoch = network.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = network.step(mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_loss_ce = step_data["loss_ce"]
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
            print ("  %sing: %d.%d / %d \t loss: %.3f = %.2f + %.2f \t avg_loss: %.3f \t"\
                   "%s \t time: %.2fs" % (mode, epoch, i * args.batch_size,
                        batches_per_epoch * args.batch_size, 
                        current_loss, current_loss_ce, current_loss_reg,
                        avg_loss / args.log_every, 
                        log, cur_time - prev_time))
            avg_loss = 0
            prev_time = cur_time
        
        if np.isnan(current_loss):
            raise Exception ("current loss IS NaN. This should never happen :)") 

        
    sum_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, sum_loss)
    metrics.print_metrics_binary(y_true, predictions)
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
    # ensuring that the code uses test_reader
    del train_reader
    del val_reader 
    
    test_reader = InHospitalMortalityReader(dataset_dir='../../data/in-hospital-mortality/test/',
                    listfile='../../data/in-hospital-mortality/test_listfile.csv',
                    period_length=48.0)

    data_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part)
    
    n_batches = len(data_raw[0]) // args.batch_size
    y_true = []
    predictions = []
    avg_loss = 0.0
    sum_loss = 0.0
    activations = []
    prev_time = time.time()

    for i in range(n_batches):
        begin_index = i * args.batch_size
        end_index = (i+1) * args.batch_size
        data = data_raw[0][begin_index:end_index]
        mortalities = data_raw[1][begin_index:end_index]

        ret = network.predict((data, mortalities))
        prediction = ret[0]
        current_loss = ret[1]
        
        avg_loss += current_loss
        sum_loss += current_loss
        
        for x in mortalities:
            y_true.append(x)
        
        for x in prediction:
            predictions.append(x)
        
        activations += zip(prediction[:, 1], mortalities)
        
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
    metrics.print_metrics_binary(y_true, predictions)
    
    with open("activations.txt", "w") as fout:
        for (x, y) in activations:
            fout.write("%.6f, %d\n" % (x, y))
else:
    raise Exception("unknown mode")
