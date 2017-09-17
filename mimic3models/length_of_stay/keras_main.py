import numpy as np
import argparse
import os
import imp
import re

from mimic3models.length_of_stay import utils
from mimic3benchmark.readers import LengthOfStayReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.set_defaults(deep_supervision=False)
parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")
args = parser.parse_args()
print args

if args.small_part:
    args.save_every = 2**30

# Build readers, discretizers, normalizers
if args.deep_supervision:
    train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir='../../data/length-of-stay/train/',
                            listfile='../../data/length-of-stay/train_listfile.csv',
                            small_part=args.small_part)
    val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir='../../data/length-of-stay/train/',
                            listfile='../../data/length-of-stay/val_listfile.csv',
                            small_part=args.small_part)
else:
    train_reader = LengthOfStayReader(dataset_dir='../../data/length-of-stay/train/',
                        listfile='../../data/length-of-stay/train_listfile.csv')
    val_reader = LengthOfStayReader(dataset_dir='../../data/length-of-stay/train/',
                        listfile='../../data/length-of-stay/val_listfile.csv')

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

if args.deep_supervision:
    discretizer_header = discretizer.transform(train_data_loader._data[0][0])[1].split(',')
else:
    discretizer_header = discretizer.transform(train_reader.read_example(0)[0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels) # choose here onlycont vs all
normalizer.load_params('los_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(args.timestep))

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'los'
args_dict['num_classes'] = (1 if args.partition == 'none' else 10)


# Build the model
print "==> using model {}".format(args.network)
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = "{}.bs{}{}{}.ts{}.partition={}".format("" if not args.deep_supervision else ".dsup",
                                                args.batch_size,
                                                ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                                ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                                args.timestep,
                                                args.partition)
model.final_name = args.prefix + model.say_name() + suffix                              
print "==> model.final_name:", model.final_name


# Compile the model
print "==> compiling the model"
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

if args.partition == 'none':
    # other options are: 'mean_squared_error', 'mean_absolute_percentage_error'
    loss_function = 'mean_squared_logarithmic_error'
else:
    loss_function = 'sparse_categorical_crossentropy'
# NOTE: categorical_crossentropy needs one-hot vectors
#       that's why we use sparse_categorical_crossentropy
# NOTE: it is ok to use keras.losses even for (B, T, D) shapes

model.compile(optimizer=optimizer_config,
              loss=loss_function)

## print model summary
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*chunk([0-9]+).*", args.load_state).group(1))


# Load data and prepare generators
if args.deep_supervision:
    train_data_gen = utils.BatchGenDeepSupervisoin(train_data_loader, args.partition,
                                        discretizer, normalizer, args.batch_size)
    val_data_gen = utils.BatchGenDeepSupervisoin(val_data_loader, args.partition,
                                        discretizer, normalizer, args.batch_size)
else:
    # Set number of batches in one epoch
    train_nbatches = 2000
    val_nbatches = 1000
    if (args.small_part):
        train_nbatches = 20
        val_nbatches = 20

    train_data_gen = utils.BatchGen(reader=train_reader,
                                    discretizer=discretizer,
                                    normalizer=normalizer,
                                    partition=args.partition,
                                    batch_size=args.batch_size,
                                    steps=train_nbatches)
    val_data_gen = utils.BatchGen(reader=val_reader,
                                discretizer=discretizer,
                                normalizer=normalizer,
                                partition=args.partition,
                                batch_size=args.batch_size,
                                steps=val_nbatches)
    #val_data_gen.steps = val_reader.get_number_of_examples() // args.batch_size
    #train_data_gen.steps = train_reader.get_number_of_examples() // args.batch_size


if args.mode == 'train':

    # Prepare training
    path = 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss}.state'
    
    metrics_callback = keras_utils.MetricsLOS(train_data_gen,
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
    # NOTE: for testing we make sure that deepsupervision is disabled
    #       and we predict examples one by one.

    # ensure that the code uses test_reader
    del train_data_gen
    del val_data_gen

    if args.deep_supervision:
        del train_data_loader
        del val_data_loader
    else:
        del train_reader
        del val_reader

    test_reader = LengthOfStayReader(dataset_dir='../../data/length-of-stay/test/',
                                    listfile='../../data/length-of-stay/test_listfile.csv')

    test_nbatches = test_reader.get_number_of_examples() // args.batch_size
    test_nbatches = 10000
    test_data_gen = utils.BatchGen(reader=test_reader,
                                discretizer=discretizer,
                                normalizer=normalizer,
                                partition=args.partition,
                                batch_size=args.batch_size,
                                steps=test_nbatches)
    labels = []
    predictions = []
    for i in range(test_nbatches):
        print "\rpredicting {} / {}".format(i, test_nbatches),
        x, y_processed, y = test_data_gen.next(return_y_true=True)
        x = np.array(x)
        pred = model.predict_on_batch(x)
        predictions += list(pred)
        labels += list(y)

    if args.partition == 'log':
        predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
        metrics.print_metrics_log_bins(labels, predictions)
    if args.partition == 'custom':
        predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
        metrics.print_metrics_custom_bins(labels, predictions)
    if args.partition == 'none':
        metrics.print_metrics_regression(labels, predictions)

    with open("activations.txt", "w") as fout:
        fout.write("prediction, y_true")
        for (x, y) in zip(predictions, labels):
            fout.write("%.6f, %.6f\n" % (x, y))

else:
    raise ValueError("Wrong value for args.mode")
