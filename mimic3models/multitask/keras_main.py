import numpy as np
import argparse
import time
import os
import imp
import re

from mimic3models.multitask import utils
from mimic3benchmark.readers import MultitaskReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")
parser.add_argument('--ihm_C', type=float, default=1.0)
parser.add_argument('--los_C', type=float, default=1.0)
parser.add_argument('--pheno_C', type=float, default=1.0)
parser.add_argument('--decomp_C', type=float, default=1.0)
args = parser.parse_args()
print args

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = MultitaskReader(dataset_dir='../../data/multitask/train/',
                            listfile='../../data/multitask/train_listfile.csv')

val_reader = MultitaskReader(dataset_dir='../../data/multitask/train/',
                            listfile='../../data/multitask/val_listfile.csv')

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)[0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels) # choose here onlycont vs all
normalizer.load_params('mult_ts%s.input_str:%s.start_time:zero.normalizer' % (args.timestep, args.imputation))

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['ihm_pos'] = int(48.0 / args.timestep - 1e-6)
args_dict['target_repl'] = target_repl

# Build the model
print "==> using model {}".format(args.network)
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
network = model # alias
suffix = ".bs{}{}{}.ts{}{}_partition={}_ihm={}_decomp={}_los={}_pheno={}".format(
                                    args.batch_size,
                                    ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                    ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                    args.timestep,
                                    ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "",
                                    args.partition,
                                    args.ihm_C,
                                    args.decomp_C,
                                    args.los_C,
                                    args.pheno_C)
model.final_name = args.prefix + model.say_name() + suffix                              
print "==> model.final_name:", model.final_name


# Compile the model
print "==> compiling the model"
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# Define loss functions

loss_dict = {}
loss_weights = {}

## ihm
if target_repl:
    loss_dict['ihm_single'] = 'binary_crossentropy'
    loss_dict['ihm_seq'] = 'binary_crossentropy'
    loss_weights['ihm_single'] = args.ihm_C * (1 - args.target_repl_coef)
    loss_weights['ihm_seq'] = args.ihm_C * args.target_repl_coef
else:
    loss_dict['ihm'] = 'binary_crossentropy'
    loss_weights['ihm'] = args.ihm_C

## decomp
loss_dict['decomp'] = 'binary_crossentropy'
loss_weights['decomp'] = args.decomp_C

## los
if args.partition == 'none':
    # other options are: 'mean_squared_error', 'mean_absolute_percentage_error'
    loss_dict['los'] = 'mean_squared_logarithmic_error'
else:
    loss_dict['los'] = 'sparse_categorical_crossentropy'
loss_weights['los'] = args.los_C

## pheno
if target_repl:
    loss_dict['pheno_single'] = 'binary_crossentropy'
    loss_dict['pheno_seq'] = 'binary_crossentropy'
    loss_weights['pheno_single'] = args.pheno_C * (1 - args.target_repl_coef)
    loss_weights['pheno_seq'] = args.pheno_C * args.target_repl_coef
else:
    loss_dict['pheno'] = 'binary_crossentropy'
    loss_weights['pheno'] = args.pheno_C

model.compile(optimizer=optimizer_config,
              loss=loss_dict,
              loss_weights=loss_weights)

## print model summary
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Build data generators
train_data_gen = utils.BatchGen(reader=train_reader,
                                discretizer=discretizer,
                                normalizer=normalizer,
                                ihm_pos=args_dict['ihm_pos'],
                                partition=args.partition,
                                target_repl=target_repl,
                                batch_size=args.batch_size,
                                small_part=args.small_part)
val_data_gen = utils.BatchGen(reader=val_reader,
                                discretizer=discretizer,
                                normalizer=normalizer,
                                ihm_pos=args_dict['ihm_pos'],
                                partition=args.partition,
                                target_repl=target_repl,
                                batch_size=args.batch_size,
                                small_part=args.small_part)

if args.mode == 'train':
    
    # Prepare training
    path = 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'

    metrics_callback = keras_utils.MetricsMultitask(train_data_gen,
                                        val_data_gen,
                                        args.partition,
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

    test_reader = MultitaskReader(dataset_dir='../../data/multitask/test/',
                              listfile='../../data/multitask/test_listfile.csv')

    test_data_gen = utils.BatchGen(reader=test_reader,
                                discretizer=discretizer,
                                normalizer=normalizer,
                                ihm_pos=args_dict['ihm_pos'],
                                partition=args.partition,
                                target_repl=target_repl,
                                batch_size=args.batch_size,
                                small_part=args.small_part)
    ihm_y_true = []
    decomp_y_true = []
    los_y_true = []
    pheno_y_true = []

    ihm_pred = []
    decomp_pred = []
    los_pred = []
    pheno_pred = []

    for i in range(test_data_gen.steps):
        print "\r\tdone {}/{}".format(i, test_data_gen.steps),
        (X, y, los_y_reg) = test_data_gen.next(return_y_true=True)
        outputs = model.predict(X, batch_size=args.batch_size)

        ihm_M = X[1]
        decomp_M = X[2]
        los_M = X[3]

        assert len(outputs) == 4 # no target replication
        (ihm_p, decomp_p, los_p, pheno_p) = outputs
        (ihm_t, decomp_t, los_t, pheno_t) = y

        los_t = los_y_reg # real value not the label

        ## ihm
        for (m, t, p) in zip(ihm_M.flatten(), ihm_t.flatten(), ihm_p.flatten()):
            if np.equal(m, 1):
                ihm_y_true.append(t)
                ihm_pred.append(p)

        ## decomp
        for (m, t, p) in zip(decomp_M.flatten(), decomp_t.flatten(), decomp_p.flatten()):
            if np.equal(m, 1):
                decomp_y_true.append(t)
                decomp_pred.append(p)

        ## los
        if los_p.shape[-1] == 1: # regression
            for (m, t, p) in zip(los_M.flatten(), los_t.flatten(), los_p.flatten()):
                if np.equal(m, 1):
                    los_y_true.append(t)
                    los_pred.append(p)
        else: # classification
            for (m, t, p) in zip(los_M.flatten(), los_t.flatten(), los_p.reshape((-1, 10))):
                if np.equal(m, 1):
                    los_y_true.append(t)
                    los_pred.append(p)

        ## pheno
        for (t, p) in zip(pheno_t.reshape((-1, 25)), pheno_p.reshape((-1, 25))):
            pheno_y_true.append(t)
            pheno_pred.append(p)
    print "\n"

    ## ihm
    if args.ihm_C > 0:
        print "\n ================= 48h mortality ================"
        ihm_pred = np.array(ihm_pred)
        ihm_pred = np.stack([1-ihm_pred, ihm_pred], axis=1)
        ihm_ret = metrics.print_metrics_binary(ihm_y_true, ihm_pred)

    ## decomp
    if args.decomp_C > 0:
        print "\n ================ decompensation ================"
        decomp_pred = np.array(decomp_pred)
        decomp_pred = np.stack([1-decomp_pred, decomp_pred], axis=1)
        decomp_ret = metrics.print_metrics_binary(decomp_y_true, decomp_pred)

    ## los
    if args.los_C > 0:
        print "\n ================ length of stay ================"
        if args.partition == 'log':
            los_pred = [metrics.get_estimate_log(x, 10) for x in los_pred]
            los_ret = metrics.print_metrics_log_bins(los_y_true, los_pred)
        if args.partition == 'custom':
            los_pred = [metrics.get_estimate_custom(x, 10) for x in los_pred]
            los_ret = metrics.print_metrics_custom_bins(los_y_true, los_pred)
        if args.partition == 'none':
            los_ret = metrics.print_metrics_regression(los_y_true, los_pred)

    ## pheno
    if args.pheno_C > 0:
        print "\n =================== phenotype =================="
        pheno_pred = np.array(pheno_pred)
        pheno_ret = metrics.print_metrics_multilabel(pheno_y_true, pheno_pred)

    # TODO: save activations if needed

elif args.mode == 'test_single':
    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_data_gen
    del val_data_gen

    # Testing ihm
    from mimic3benchmark.readers import InHospitalMortalityReader
    from mimic3models.in_hospital_mortality.utils import read_chunk
    from mimic3models import nn_utils

    test_reader = InHospitalMortalityReader(dataset_dir='../../data/in-hospital-mortality/test/',
                    listfile='../../data/in-hospital-mortality/test_listfile.csv',
                    period_length=48.0)

    ihm_y_true = []
    ihm_pred = []

    nsteps = test_reader.get_number_of_examples() // args.batch_size
    for iteration in range(nsteps):
        (X, ts, labels, header) = read_chunk(test_reader, args.batch_size)

        for i in range(args.batch_size):
            X[i] = discretizer.transform(X[i], end=48.0)[0]
            X[i] = normalizer.transform(X[i])

        X = nn_utils.pad_zeros(X, min_length=args_dict['ihm_pos']+1)
        T = X.shape[1]
        ihm_M = np.ones(shape=(args.batch_size,1))
        decomp_M = np.ones(shape=(args.batch_size, T))
        los_M = np.ones(shape=(args.batch_size, T))

        pred = model.predict([X, ihm_M, decomp_M, los_M])[0]
        ihm_y_true += labels
        ihm_pred += list(pred.flatten())

    print "\n ================= 48h mortality ================"
    ihm_pred = np.array(ihm_pred)
    ihm_pred = np.stack([1-ihm_pred, ihm_pred], axis=1)
    ihm_ret = metrics.print_metrics_binary(ihm_y_true, ihm_pred)

else:
    raise ValueError("Wrong value for args.mode")
