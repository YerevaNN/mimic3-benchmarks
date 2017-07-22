import numpy as np
import argparse
import time
import os
import imp
import re

from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
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
args_dict['task'] = 'decomp'


# Build the model
print "==> using model {}".format(args.network)
model_module = imp.load_source(os.path.basename(args.network), args.network)
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
    path = 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss}.state'
    
    metrics_callback = keras_utils.MetricsBinaryFromGenerator(train_data_gen,
                                                            val_data_gen,
                                                            args.batch_size)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)
    
    if not os.path.exists('keras_logs'):
        os.makedirs('keras_logs')
    csv_logger = CSVLogger(os.path.join('keras_logs', model.final_name + '.csv'),
                           append=True, separator=';')
    
    print "==> training"
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=train_data_gen.steps,
                        validation_data=val_data_gen,
                        validation_steps=val_data_gen.steps,
                        epochs=args.epochs,
                        initial_epoch=n_trained_chunks,
                        callbacks=[metrics_callback, saver, csv_logger])

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_data_gen
    del val_data_gen
    
    test_reader = DecompensationReader(dataset_dir='../../data/decompensation/test/',
            listfile='../../data/decompensation/test_listfile.csv')
    
    test_nbatches = test_reader.get_number_of_examples() // args.batch_size
    test_nbatches = 10000
    test_data_gen = utils.BatchGen(test_reader, discretizer,
                                    normalizer, args.batch_size,
                                    test_nbatches)
    labels = []
    predictions = []
    for i in range(test_nbatches):
        print "\rpredicting {} / {}".format(i, test_nbatches),
        x, y = next(test_data_gen)
        x = np.array(x)
        pred = model.predict_on_batch(x)[:, 0]
        predictions += list(pred)
        labels += list(y)
    
    metrics.print_metrics_binary(labels, predictions)
    with open("activations.txt", "w") as fout:
        fout.write("predictions, labels\n")
        for (x, y) in zip(predictions, labels):
            fout.write("%.6f, %d\n" % (x, y))

else:
    raise ValueError("Wrong value for args.mode")
