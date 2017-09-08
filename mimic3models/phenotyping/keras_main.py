import numpy as np
import argparse
import time
import os
import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
args = parser.parse_args()
print args

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = PhenotypingReader(dataset_dir='../../data/phenotyping/train/',
                                listfile='../../data/phenotyping/train_listfile.csv')

val_reader = PhenotypingReader(dataset_dir='../../data/phenotyping/train/',
                                listfile='../../data/phenotyping/val_listfile.csv')

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)[0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels) # choose here onlycont vs all
normalizer.load_params('ph_ts%s.input_str:previous.start_time:zero.normalizer' % args.timestep)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ph'
args_dict['num_classes'] = 25
args_dict['target_repl'] = target_repl

# Build the model
print "==> using model {}".format(args.network)
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
network = model # alias
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix                              
print "==> model.final_name:", model.final_name


# Compile the model
print "==> compiling the model"
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)

## print model summary
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Build data generators
train_data_gen = utils.BatchGen(train_reader, discretizer,
                                normalizer, args.batch_size,
                                args.small_part, target_repl)
val_data_gen = utils.BatchGen(val_reader, discretizer,
                              normalizer, args.batch_size,
                              args.small_part, target_repl)

if args.mode == 'train':
    
    # Prepare training
    path = 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'
    
    metrics_callback = keras_utils.MetricsMultilabel(train_data_gen,
                                                   val_data_gen,
                                                   args.batch_size,
                                                   args.verbose)
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
                        epochs=n_trained_chunks + args.epochs,
                        initial_epoch=n_trained_chunks,
                        callbacks=[metrics_callback, saver, csv_logger],
                        verbose=args.verbose)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_data_gen
    del val_data_gen
    
    test_reader = PhenotypingReader(dataset_dir='../../data/phenotyping/test/',
                    listfile='../../data/phenotyping/test_listfile.csv')
    
    test_data_gen = utils.BatchGen(test_reader, discretizer,
                                    normalizer, args.batch_size,
                                    args.small_part, target_repl)
    test_nbatches = test_data_gen.steps
    #test_nbatches = 2

    labels = []
    predictions = []
    for i in range(test_nbatches):
        print "\rpredicting {} / {}".format(i, test_nbatches),
        x, y = next(test_data_gen)
        x = np.array(x)
        pred = model.predict_on_batch(x)
        predictions += list(pred)
        labels += list(y)

    ret = metrics.print_metrics_multilabel(labels, predictions)
    
    with open("results.txt", "w") as resfile:
        header = "ave_prec_micro,ave_prec_macro,ave_prec_weighted,"
        header += "ave_recall_micro,ave_recall_macro,ave_recall_weighted,"
        header += "ave_auc_micro,ave_auc_macro,ave_auc_weighted,"
        header += ','.join(["auc_%d" % i for i in range(args_dict['num_classes'])])
        resfile.write(header + "\n")
        
        resfile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f," % (
            ret['ave_prec_micro'], ret['ave_prec_macro'], ret['ave_prec_weighted'],
            ret['ave_recall_micro'], ret['ave_recall_macro'], ret['ave_recall_weighted'],
            ret['ave_auc_micro'], ret['ave_auc_macro'], ret['ave_auc_weighted']))
        resfile.write(",".join(["%.6f" % x for x in ret['auc_scores']]) + "\n")
    
    np.savetxt("activations.csv", predictions, delimiter=',')
    np.savetxt("answer.csv", np.array(labels, dtype=np.int32), delimiter=',')

else:
    raise ValueError("Wrong value for args.mode")
