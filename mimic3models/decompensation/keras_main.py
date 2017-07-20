import numpy as np
import argparse
import time
import os
import importlib
import re

from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader

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
train_reader = DecompensationReader(dataset_dir='../../data/decompensation/train/',
                    listfile='../../data/decompensation/train_listfile.csv')

val_reader = DecompensationReader(dataset_dir='../../data/decompensation/train/',
                    listfile='../../data/decompensation/val_listfile.csv')

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)[0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels) # choose here onlycont vs all
normalizer.load_params('decomp_ts0.8.input_str:previous.n1e5.start_time:zero.normalizer')

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
    n_trained_chunks = 1 + int(re.match(".*chunk([0-9]+).*", args.load_state).group(1))


# Set number of batches in one epoch
train_nbatches = 2000
val_nbatches = 1000

if (args.small_part):
    train_nbatches = 20
    val_nbatches = 20
    args.save_every = 2**30


# Build data generators
train_data_gen = utils.BatchGen(train_reader, discretizer,
                                normalizer, args.batch_size, train_nbatches)
val_data_gen = utils.BatchGen(val_reader, discretizer,
                              normalizer, args.batch_size, val_nbatches)
#train_data_gen.steps = train_reader.get_number_of_examples() / args.batch_size
#val_data_gen.steps = val_reader.get_number_of_examples() / args.batch_size



if args.mode == 'train':
    
    # Prepare training
    print "==> training"
    # TODO: write callback for model save
    path = 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss}.state'
    
    metrics_binary = keras_utils.MetricsBinaryFromGenerator(train_data_gen,
                                                            val_data_gen,
                                                            args.batch_size)
    
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)
    
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=train_data_gen.steps,
                        validation_data=val_data_gen,
                        validation_steps=val_data_gen.steps,
                        epochs=args.epochs,
                        initial_epoch=n_trained_chunks,
                        callbacks=[metrics_binary, saver])

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_data_gen
    del val_data_gen
    
    test_reader = DecompensationReader(dataset_dir='../../data/decompensation/test/',
            listfile='../../data/decompensation/test_listfile.csv')
    
    test_nbatches = test_reader.get_number_of_examples() // args.batch_size
    test_data_gen = utils.BatchGen(test_reader, discretizer,
                                   normalizer, args.batch_size, test_nbatches)
    
    # TODO: write the remaining part later :)
    # NOTE: you may want to store the activation in one file, line this
    """
    with open("activations.txt", "w") as fout:
        for (x, y) in activations:
            fout.write("%.6f, %d\n" % (x, y))
    """
    
else:
    raise ValueError("Wrong value for args.mode")
