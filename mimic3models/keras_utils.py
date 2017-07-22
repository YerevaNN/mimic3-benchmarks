import numpy as np
import metrics

import keras
import keras.backend as K
from keras.layers import Layer


# ===================== METRICS ===================== #                        

class MetricsBinaryFromGenerator(keras.callbacks.Callback):
    
    def __init__(self, train_data_gen, val_data_gen, batch_size=32):
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            print "\r\tdone {}/{}".format(i, data_gen.steps),
            (x,y) = next(data_gen)
            y_true += list(y)
            predictions += list(self.model.predict(x, batch_size=self.batch_size))
        print "\n"
        predictions = np.array(predictions)
        predictions = np.stack([1-predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        for k, v in ret.iteritems():
            logs[dataset + '_' + k] = v
        history.append(ret)
    
    def on_epoch_end(self, epoch, logs={}):
        print "\n==>predicting on train"
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print "\n==>predicting on validation"
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)


class MetricsBinaryFromData(keras.callbacks.Callback):
    
    def __init__(self, train_data, val_data, batch_size=32):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data, history, dataset, logs):
        y_true = []
        predictions = []
        num_examples = len(data[0])
        for i in range(0, num_examples, self.batch_size):
            print "\r\tdone {}/{}".format(i, num_examples),
            (x,y) = (data[0][i:i+self.batch_size], data[1][i:i+self.batch_size])
            y_true += list(y)
            predictions += list(self.model.predict(x, batch_size=self.batch_size))
        print "\n"
        predictions = np.array(predictions)
        predictions = np.stack([1-predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        for k, v in ret.iteritems():
            logs[dataset + '_' + k] = v
        history.append(ret)
        
    def on_epoch_end(self, epoch, logs={}):
        print "\n==>predicting on train"
        self.calc_metrics(self.train_data, self.train_history, 'train', logs)
        print "\n==>predicting on validation"
        self.calc_metrics(self.val_data, self.val_history, 'val', logs)
                        

class MetricsMultilabel(keras.callbacks.Callback):
    
    def __init__(self, train_data_gen, val_data_gen, batch_size=32):
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            print "\r\tdone {}/{}".format(i, data_gen.steps),
            (x, y) = next(data_gen)
            y_true += list(y)
            predictions += list(self.model.predict(x, batch_size=self.batch_size))
        print "\n"
        predictions = np.array(predictions)
        ret = metrics.print_metrics_multilabel(y_true, predictions)
        for k, v in ret.iteritems():
            logs[dataset + '_' + k] = v
        history.append(ret)
    
    def on_epoch_end(self, epoch, logs={}):
        print "\n==>predicting on train"
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print "\n==>predicting on validation"
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)


class MetricsLOS(keras.callbacks.Callback):
    
    def __init__(self, train_data_gen, val_data_gen, partition, batch_size=32):
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.partition = partition
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            print "\r\tdone {}/{}".format(i, data_gen.steps),
            (x,y) = next(data_gen)
            y_true += list(y)
            predictions += list(self.model.predict(x, batch_size=self.batch_size))
        print "\n"
        predictions = np.array(predictions)
        
        if self.partition == 'log':
            predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
            ret = metrics.print_metrics_log_bins(y_true, predictions)
        if self.partition == 'custom':
            predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
            ret = metrics.print_metrics_custom_bins(y_true, predictions)
        if self.partition == 'none':
            ret = metrics.print_metrics_regression(y_true, predictions)
        for k, v in ret.iteritems():
            logs[dataset + '_' + k] = v
        history.append(ret)
    
    def on_epoch_end(self, epoch, logs={}):
        print "\n==>predicting on train"
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print "\n==>predicting on validation"
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)


# ===================== LAYERS ===================== #                        

class CollectAttetion(Layer):
    """ Collect attention on 3D tensor with softmax and summation
        Masking is disabled after this layer
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(CollectAttetion, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        x = inputs[0]
        a = inputs[1]
        
        # softmax attention
        if mask is not None:
            # 2 masks coming from incoming layers, both are the same
            # each mask is 2D (batch_size, timestep)
            a = a * K.expand_dims(mask[0], axis=2)
        
        a = K.squeeze(a, axis=2)
        a = K.softmax(a)
        a = K.expand_dims(a, axis=2)

        # collect xs
        return K.sum(x * a, axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][2]

    def compute_mask(self, input, input_mask=None):
        return None


class Slice(Layer):
    """ Slice 3D tensor by taking x[:, :, indices]
    """
    def __init__(self, indices, **kwargs):
        self.supports_masking = True
        self.indices = indices
        super(Slice, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, self.indices]
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], len(self.indices)

    def compute_mask(self, input, input_mask=None):
        return input_mask
