import numpy as np
import metrics

import keras
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

from keras.layers import Layer, LSTM
from keras.layers.recurrent import _time_distributed_dense

# ===================== METRICS ===================== #                        

class MetricsBinaryFromGenerator(keras.callbacks.Callback):
    
    def __init__(self, train_data_gen, val_data_gen, batch_size=32, verbose=2):
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.verbose = verbose
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print "\r\tdone {}/{}".format(i, data_gen.steps),
            (x,y) = next(data_gen)
            pred = self.model.predict(x, batch_size=self.batch_size)

            if isinstance(x, list) and len(x) == 2: # deep supervision
                masks = x[1]
                for smask, sy, spred in zip(masks, y, pred): # examples
                    for tm, ty, tp in zip(smask, sy, spred): # timesteps
                        if np.int(tm) == 1:
                            y_true.append(np.int(ty))
                            predictions.append(np.float(tp))
            else:
                y_true += list(y.flatten())
                predictions += list(pred.flatten())
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
    
    def __init__(self, train_data, val_data, batch_size=32, verbose=2):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.verbose = verbose
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data, history, dataset, logs):
        y_true = []
        predictions = []
        num_examples = len(data[0])
        for i in range(0, num_examples, self.batch_size):
            if self.verbose == 1:
                print "\r\tdone {}/{}".format(i, num_examples),
            (x,y) = (data[0][i:i+self.batch_size], data[1][i:i+self.batch_size])
            if len(y) == 2: # target replication
                y_true += list(y[0].flatten())
            else:
                y_true += list(y.flatten())
            outputs = self.model.predict(x, batch_size=self.batch_size)
            if len(outputs) == 2:
                predictions += list(outputs[0].flatten())
            else:
                predictions += list(outputs.flatten())
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
    
    def __init__(self, train_data_gen, val_data_gen, batch_size=32, verbose=2):
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.verbose = verbose
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print "\r\tdone {}/{}".format(i, data_gen.steps),
            (x, y) = next(data_gen)
            if len(y) == 2:
                y_true += list(y[0])
            else:
                y_true += list(y)
            outputs = self.model.predict(x, batch_size=self.batch_size)
            if len(outputs) == 2:
                predictions += list(outputs[0])
            else:
                predictions += list(outputs)
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
    
    def __init__(self, train_data_gen, val_data_gen, partition, batch_size=32, verbose=2):
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.partition = partition
        self.verbose = verbose
    
    def on_train_begin(self, logs={}):
        self.train_history = []
        self.val_history = []
    
    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print "\r\tdone {}/{}".format(i, data_gen.steps),
            (x, y_processed) = next(data_gen)
            y = data_gen.last_y_true # true length of stays (not bins !)
            pred = self.model.predict(x, batch_size=self.batch_size)

            if isinstance(x, list) and len(x) == 2: # deep supervision
                masks = x[1]
                for smask, sy, spred in zip(masks, y, pred): # examples
                    for tm, ty, tp in zip(smask, sy, spred): # timesteps
                        if np.int(tm) == 1:
                            y_true.append(np.float(ty))
                            if len(tp) == 1: # none
                                predictions.append(np.float(tp))
                            else: # custom or log
                                predictions.append(tp)
            else:
                if y.shape[-1] == 1:
                    y_true += list(y.flatten())
                    predictions += list(pred.flatten())
                else:
                    y_true += list(y)
                    predictions += list(pred)
        print "\n"

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

def softmax(x, axis, mask=None):
    if mask is None:
        mask = K.constant(True)
    mask = K.cast(mask, K.floatx())
    if K.ndim(x) is K.ndim(mask) + 1:
        mask = K.expand_dims(mask)

    m = K.max(x, axis=axis, keepdims=True)
    e = K.exp(x - m) * mask
    s = K.sum(e, axis=axis, keepdims=True)
    s += K.cast(K.cast(s < K.epsilon(), K.floatx()) * K.epsilon(), K.floatx())
    return e / s


def _collect_attention(x, a, mask):
    """
    x is (B, T, D)
    a is (B, T, 1) or (B, T)
    mask is (B, T)
    """
    if K.ndim(a) == 2:
        a = K.expand_dims(a)
    a = softmax(a, axis=1, mask=mask) # (B, T, 1)
    return K.sum(x * a, axis=1) # (B, D)


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
        # mask has 2 components, both are the same
        return _collect_attention(x, a, mask[0])
    
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
        if K.backend() == 'tensorflow':
            xt = tf.transpose(x, perm=(2, 0 ,1))
            gt = tf.gather(xt, self.indices)
            return tf.transpose(gt, perm=(1, 2, 0))
        return x[:, :, self.indices]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.indices))

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def get_config(self):
        return {'indices': self.indices}


class LastTimestep(Layer):
    """ Takes 3D tensor and returns x[:, -1, :]
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(LastTimestep, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # TODO: test on tensorflow
        return x[:, -1, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, input, input_mask=None):
        return None


class ExtendMask(Layer):
    """ Inputs:      [X, M]
        Output:      X
        Output_mask: M
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(ExtendMask, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, input, input_mask=None):
        return K.squeeze(input[1], axis=-1)
