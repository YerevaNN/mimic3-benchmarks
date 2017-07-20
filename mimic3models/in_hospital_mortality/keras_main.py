import numpy as np
import argparse
import time
import os
import importlib
import re

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils

from keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, required=True)
parser.add_argument('--dim', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--depth', type=int, default=0,
                    help='number of bi-LSTMs')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of chunks to train')
parser.add_argument('--load_state', type=str, default="",
                    help='state file path')
parser.add_argument('--mode', type=str, default="train",
                    help='mode: train or test')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
parser.add_argument('--save_every', type=int, default=1,
                    help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="",
                    help='optional prefix of network name')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--batch_norm', type=bool, default=False,
                    help='batch normalization')
parser.add_argument('--timestep', type=float, default=0.8,
                    help="fixed timestep used in the dataset")
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--whole_data', dest='small_part', action='store_false')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9,
                    help='beta_1 param for Adam optimizer')
parser.set_defaults(small_part=False)
args = parser.parse_args()
print args


# Build readers, discretizers, normalizers
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

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header


# Build the model
print "==> using model {}".format(args.network)
model_module = importlib.import_module(args.network.replace('/', '.')
                                                   .replace('.py', ''))
model = model_module.Network(**args_dict)
network = model # alias
suffix = ".bs{}{}{}.ts{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep)
model.final_name = args.prefix + model.say_name() + suffix                              
print "==> model.final_name:", model.final_name


# Compile the model
print "==> compiling the model"
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

model.compile(optimizer=optimizer_config,
              loss='binary_crossentropy',
              metrics=['accuracy'])

## print model summary
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = 1 + int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Read data
train_raw = utils.load_mortalities(train_reader, discretizer, normalizer, args.small_part)
val_raw = utils.load_mortalities(val_reader, discretizer, normalizer, args.small_part)
if args.small_part:
    args.save_every = 2**30


if args.mode == 'train':
    
    # Prepare training
    print "==> training"
    # TODO: write callback for model save
    path = 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'
    
    metrics_binary = keras_utils.MetricsBinaryFromData(train_raw,
                                                       val_raw,
                                                       args.batch_size)
    
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)
    
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_binary, saver])


elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw
    
    test_reader = InHospitalMortalityReader(dataset_dir='../../data/in-hospital-mortality/test/',
                    listfile='../../data/in-hospital-mortality/test_listfile.csv',
                    period_length=48.0)
    test_raw = utils.load_mortalities(test_reader, discretizer, normalizer, args.small_part)
    
    data = np.array(test_raw[0])
    mortalities = test_raw[1]
    predictions = model.predict(data,
                                batch_size=args.batch_size,
                                verbose=2)
    predictions = np.array(predictions)[:, 0]
    activations = zip(predictions, mortalities)

    metrics.print_metrics_binary(mortalities, predictions)
    with open("activations.txt", "w") as fout:
        for (x, y) in activations:
            fout.write("%.6f, %d\n" % (x, y))

else:
    raise ValueError("Wrong value for args.mode")
