from mimic3models.multitask import utils
from mimic3benchmark.readers import MultitaskReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
from keras.callbacks import ModelCheckpoint, CSVLogger

import mimic3models.in_hospital_mortality.utils as ihm_utils
import mimic3models.decompensation.utils as decomp_utils
import mimic3models.length_of_stay.utils as los_utils
import mimic3models.phenotyping.utils as pheno_utils

import numpy as np
import argparse
import time
import os
import imp
import re

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--partition', type=str, default='custom', help="log, custom, none")
parser.add_argument('--ihm_C', type=float, default=1.0)
parser.add_argument('--los_C', type=float, default=1.0)
parser.add_argument('--pheno_C', type=float, default=1.0)
parser.add_argument('--decomp_C', type=float, default=1.0)
args = parser.parse_args()
print args

if args.small_part:
    args.save_every = 2 ** 30

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

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all
normalizer.load_params('mult_ts%s.input_str:%s.start_time:zero.normalizer' % (args.timestep, args.imputation))

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['ihm_pos'] = int(48.0 / args.timestep - 1e-6)
args_dict['target_repl'] = target_repl

# Build the model
print "==> using model {}".format(args.network)
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
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

# ihm
if target_repl:
    loss_dict['ihm_single'] = 'binary_crossentropy'
    loss_dict['ihm_seq'] = 'binary_crossentropy'
    loss_weights['ihm_single'] = args.ihm_C * (1 - args.target_repl_coef)
    loss_weights['ihm_seq'] = args.ihm_C * args.target_repl_coef
else:
    loss_dict['ihm'] = 'binary_crossentropy'
    loss_weights['ihm'] = args.ihm_C

# decomp
loss_dict['decomp'] = 'binary_crossentropy'
loss_weights['decomp'] = args.decomp_C

# los
if args.partition == 'none':
    # other options are: 'mean_squared_error', 'mean_absolute_percentage_error'
    loss_dict['los'] = 'mean_squared_logarithmic_error'
else:
    loss_dict['los'] = 'sparse_categorical_crossentropy'
loss_weights['los'] = args.los_C

# pheno
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
                                small_part=args.small_part,
                                shuffle=True)
val_data_gen = utils.BatchGen(reader=val_reader,
                              discretizer=discretizer,
                              normalizer=normalizer,
                              ihm_pos=args_dict['ihm_pos'],
                              partition=args.partition,
                              target_repl=target_repl,
                              batch_size=args.batch_size,
                              small_part=args.small_part,
                              shuffle=False)

if args.mode == 'train':
    # Prepare training
    path = 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'

    metrics_callback = keras_utils.MultitaskMetrics(train_data_gen=train_data_gen,
                                                    val_data_gen=val_data_gen,
                                                    partition=args.partition,
                                                    batch_size=args.batch_size,
                                                    verbose=args.verbose)
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
                                   small_part=args.small_part,
                                   shuffle=False,
                                   return_names=True)
    ihm_y_true = []
    decomp_y_true = []
    los_y_true = []
    pheno_y_true = []

    ihm_pred = []
    decomp_pred = []
    los_pred = []
    pheno_pred = []

    ihm_names = []
    decomp_names = []
    los_names = []
    pheno_names = []

    decomp_ts = []
    los_ts = []
    pheno_ts = []

    for i in range(test_data_gen.steps):
        print "\r\tdone {}/{}".format(i, test_data_gen.steps),
        ret = test_data_gen.next(return_y_true=True)
        (X, y, los_y_reg) = ret["data"]
        outputs = model.predict(X, batch_size=args.batch_size)

        names = list(ret["names"])
        names_extended = np.array(names).repeat(X[0].shape[1], axis=-1)

        ihm_M = X[1]
        decomp_M = X[2]
        los_M = X[3]

        assert len(outputs) == 4  # no target replication
        (ihm_p, decomp_p, los_p, pheno_p) = outputs
        (ihm_t, decomp_t, los_t, pheno_t) = y

        los_t = los_y_reg  # real value not the label

        # ihm
        for (m, t, p, name) in zip(ihm_M.flatten(), ihm_t.flatten(), ihm_p.flatten(), names):
            if np.equal(m, 1):
                ihm_y_true.append(t)
                ihm_pred.append(p)
                ihm_names.append(name)

        # decomp
        for x in ret['decomp_ts']:
            decomp_ts += x
        for (name, m, t, p) in zip(names_extended.flatten(), decomp_M.flatten(),
                                   decomp_t.flatten(), decomp_p.flatten()):
            if np.equal(m, 1):
                decomp_names.append(name)
                decomp_y_true.append(t)
                decomp_pred.append(p)

        # los
        for x in ret['los_ts']:
            los_ts += x
        if los_p.shape[-1] == 1:  # regression
            for (name, m, t, p) in zip(names_extended.flatten(), los_M.flatten(),
                                       los_t.flatten(), los_p.flatten()):
                if np.equal(m, 1):
                    los_names.append(name)
                    los_y_true.append(t)
                    los_pred.append(p)
        else:  # classification
            for (name, m, t, p) in zip(names_extended.flatten(), los_M.flatten(),
                                       los_t.flatten(), los_p.reshape((-1, 10))):
                if np.equal(m, 1):
                    los_names.append(name)
                    los_y_true.append(t)
                    los_pred.append(p)

        # pheno
        pheno_names += list(names)
        pheno_ts += list(ret["pheno_ts"])
        for (t, p) in zip(pheno_t.reshape((-1, 25)), pheno_p.reshape((-1, 25))):
            pheno_y_true.append(t)
            pheno_pred.append(p)
    print "\n"

    # ihm
    if args.ihm_C > 0:
        print "\n ================= 48h mortality ================"
        ihm_pred = np.array(ihm_pred)
        ihm_ret = metrics.print_metrics_binary(ihm_y_true, ihm_pred)

    # decomp
    if args.decomp_C > 0:
        print "\n ================ decompensation ================"
        decomp_pred = np.array(decomp_pred)
        decomp_ret = metrics.print_metrics_binary(decomp_y_true, decomp_pred)

    # los
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

    # pheno
    if args.pheno_C > 0:
        print "\n =================== phenotype =================="
        pheno_pred = np.array(pheno_pred)
        pheno_ret = metrics.print_metrics_multilabel(pheno_y_true, pheno_pred)

    print "Saving the predictions in test_predictions/task directories ..."

    # ihm
    ihm_path = os.path.join("test_predictions/ihm", os.path.basename(args.load_state)) + ".csv"
    ihm_utils.save_results(ihm_names, ihm_pred, ihm_y_true, ihm_path)

    # decomp
    decomp_path = os.path.join("test_predictions/decomp", os.path.basename(args.load_state)) + ".csv"
    decomp_utils.save_results(decomp_names, decomp_ts, decomp_pred, decomp_y_true, decomp_path)

    # los
    los_path = os.path.join("test_predictions/los", os.path.basename(args.load_state)) + ".csv"
    los_utils.save_results(los_names, los_ts, los_pred, los_y_true, los_path)

    # pheno
    pheno_path = os.path.join("test_predictions/pheno", os.path.basename(args.load_state)) + ".csv"
    pheno_utils.save_results(pheno_names, pheno_ts, pheno_pred, pheno_y_true, pheno_path)

else:
    raise ValueError("Wrong value for args.mode")
