from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from mimic3models import metrics

import keras
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

from keras.layers import Layer


# ===================== METRICS ===================== #


class DecompensationMetrics(keras.callbacks.Callback):
    def __init__(self, train_data_gen, val_data_gen, deep_supervision,
                 batch_size=32, early_stopping=True, verbose=2):
        super(DecompensationMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.deep_supervision = deep_supervision
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (x, y) = next(data_gen)
            pred = self.model.predict(x, batch_size=self.batch_size)
            if self.deep_supervision:
                for m, t, p in zip(x[1].flatten(), y.flatten(), pred.flatten()):
                    if np.equal(m, 1):
                        y_true.append(t)
                        predictions.append(p)
            else:
                y_true += list(y.flatten())
                predictions += list(pred.flatten())
        print('\n')
        predictions = np.array(predictions)
        predictions = np.stack([1 - predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            max_auc = np.max([x["auroc"] for x in self.val_history])
            cur_auc = self.val_history[-1]["auroc"]
            if max_auc > 0.88 and cur_auc < 0.86:
                self.model.stop_training = True


class InHospitalMortalityMetrics(keras.callbacks.Callback):
    def __init__(self, train_data, val_data, target_repl, batch_size=32, early_stopping=True, verbose=2):
        super(InHospitalMortalityMetrics, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.target_repl = target_repl
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data, history, dataset, logs):
        y_true = []
        predictions = []
        B = self.batch_size
        for i in range(0, len(data[0]), B):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, len(data[0])), end='\r')
            if self.target_repl:
                (x, y, y_repl) = (data[0][i:i + B], data[1][0][i:i + B], data[1][1][i:i + B])
            else:
                (x, y) = (data[0][i:i + B], data[1][i:i + B])
            outputs = self.model.predict(x, batch_size=B)
            if self.target_repl:
                predictions += list(np.array(outputs[0]).flatten())
            else:
                predictions += list(np.array(outputs).flatten())
            y_true += list(np.array(y).flatten())
        print('\n')
        predictions = np.array(predictions)
        predictions = np.stack([1 - predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data, self.val_history, 'val', logs)

        if self.early_stopping:
            max_auc = np.max([x["auroc"] for x in self.val_history])
            cur_auc = self.val_history[-1]["auroc"]
            if max_auc > 0.85 and cur_auc < 0.83:
                self.model.stop_training = True


class PhenotypingMetrics(keras.callbacks.Callback):
    def __init__(self, train_data_gen, val_data_gen, batch_size=32,
                 early_stopping=True, verbose=2):
        super(PhenotypingMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (x, y) = next(data_gen)
            outputs = self.model.predict(x, batch_size=self.batch_size)
            if data_gen.target_repl:
                y_true += list(y[0])
                predictions += list(outputs[0])
            else:
                y_true += list(y)
                predictions += list(outputs)
        print('\n')
        predictions = np.array(predictions)
        ret = metrics.print_metrics_multilabel(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            max_auc = np.max([x["ave_auc_macro"] for x in self.val_history])
            cur_auc = self.val_history[-1]["ave_auc_macro"]
            if max_auc > 0.75 and cur_auc < 0.73:
                self.model.stop_training = True


class LengthOfStayMetrics(keras.callbacks.Callback):
    def __init__(self, train_data_gen, val_data_gen, partition, batch_size=32,
                 early_stopping=True, verbose=2):
        super(LengthOfStayMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.partition = partition
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (x, y_processed, y) = data_gen.next(return_y_true=True)
            pred = self.model.predict(x, batch_size=self.batch_size)
            if isinstance(x, list) and len(x) == 2:  # deep supervision
                if pred.shape[-1] == 1:  # regression
                    pred_flatten = pred.flatten()
                else:  # classification
                    pred_flatten = pred.reshape((-1, 10))
                for m, t, p in zip(x[1].flatten(), y.flatten(), pred_flatten):
                    if np.equal(m, 1):
                        y_true.append(t)
                        predictions.append(p)
            else:
                if pred.shape[-1] == 1:
                    y_true += list(y.flatten())
                    predictions += list(pred.flatten())
                else:
                    y_true += list(y)
                    predictions += list(pred)
        print('\n')
        if self.partition == 'log':
            predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
            ret = metrics.print_metrics_log_bins(y_true, predictions)
        if self.partition == 'custom':
            predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
            ret = metrics.print_metrics_custom_bins(y_true, predictions)
        if self.partition == 'none':
            ret = metrics.print_metrics_regression(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            max_kappa = np.max([x["kappa"] for x in self.val_history])
            cur_kappa = self.val_history[-1]["kappa"]
            max_train_kappa = np.max([x["kappa"] for x in self.train_history])
            if max_kappa > 0.38 and cur_kappa < 0.35 and max_train_kappa > 0.47:
                self.model.stop_training = True


class MultitaskMetrics(keras.callbacks.Callback):
    def __init__(self, train_data_gen, val_data_gen, partition,
                 batch_size=32, early_stopping=True, verbose=2):
        super(MultitaskMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.partition = partition
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        ihm_y_true = []
        decomp_y_true = []
        los_y_true = []
        pheno_y_true = []

        ihm_pred = []
        decomp_pred = []
        los_pred = []
        pheno_pred = []

        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (X, y, los_y_reg) = data_gen.next(return_y_true=True)
            outputs = self.model.predict(X, batch_size=self.batch_size)

            ihm_M = X[1]
            decomp_M = X[2]
            los_M = X[3]

            if not data_gen.target_repl:  # no target replication
                (ihm_p, decomp_p, los_p, pheno_p) = outputs
                (ihm_t, decomp_t, los_t, pheno_t) = y
            else:  # target replication
                (ihm_p, _, decomp_p, los_p, pheno_p, _) = outputs
                (ihm_t, _, decomp_t, los_t, pheno_t, _) = y

            los_t = los_y_reg  # real value not the label

            # ihm
            for (m, t, p) in zip(ihm_M.flatten(), ihm_t.flatten(), ihm_p.flatten()):
                if np.equal(m, 1):
                    ihm_y_true.append(t)
                    ihm_pred.append(p)

            # decomp
            for (m, t, p) in zip(decomp_M.flatten(), decomp_t.flatten(), decomp_p.flatten()):
                if np.equal(m, 1):
                    decomp_y_true.append(t)
                    decomp_pred.append(p)

            # los
            if los_p.shape[-1] == 1:  # regression
                for (m, t, p) in zip(los_M.flatten(), los_t.flatten(), los_p.flatten()):
                    if np.equal(m, 1):
                        los_y_true.append(t)
                        los_pred.append(p)
            else:  # classification
                for (m, t, p) in zip(los_M.flatten(), los_t.flatten(), los_p.reshape((-1, 10))):
                    if np.equal(m, 1):
                        los_y_true.append(t)
                        los_pred.append(p)

            # pheno
            for (t, p) in zip(pheno_t.reshape((-1, 25)), pheno_p.reshape((-1, 25))):
                pheno_y_true.append(t)
                pheno_pred.append(p)
        print('\n')

        # ihm
        print("\n ================= 48h mortality ================")
        ihm_pred = np.array(ihm_pred)
        ihm_pred = np.stack([1 - ihm_pred, ihm_pred], axis=1)
        ret = metrics.print_metrics_binary(ihm_y_true, ihm_pred)
        for k, v in ret.items():
            logs[dataset + '_ihm_' + k] = v

        # decomp
        print("\n ================ decompensation ================")
        decomp_pred = np.array(decomp_pred)
        decomp_pred = np.stack([1 - decomp_pred, decomp_pred], axis=1)
        ret = metrics.print_metrics_binary(decomp_y_true, decomp_pred)
        for k, v in ret.items():
            logs[dataset + '_decomp_' + k] = v

        # los
        print("\n ================ length of stay ================")
        if self.partition == 'log':
            los_pred = [metrics.get_estimate_log(x, 10) for x in los_pred]
            ret = metrics.print_metrics_log_bins(los_y_true, los_pred)
        if self.partition == 'custom':
            los_pred = [metrics.get_estimate_custom(x, 10) for x in los_pred]
            ret = metrics.print_metrics_custom_bins(los_y_true, los_pred)
        if self.partition == 'none':
            ret = metrics.print_metrics_regression(los_y_true, los_pred)
        for k, v in ret.items():
            logs[dataset + '_los_' + k] = v

        # pheno
        print("\n =================== phenotype ==================")
        pheno_pred = np.array(pheno_pred)
        ret = metrics.print_metrics_multilabel(pheno_y_true, pheno_pred)
        for k, v in ret.items():
            logs[dataset + '_pheno_' + k] = v

        history.append(logs)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            ihm_max_auc = np.max([x["val_ihm_auroc"] for x in self.val_history])
            ihm_cur_auc = self.val_history[-1]["val_ihm_auroc"]
            pheno_max_auc = np.max([x["val_pheno_ave_auc_macro"] for x in self.val_history])
            pheno_cur_auc = self.val_history[-1]["val_pheno_ave_auc_macro"]
            if (pheno_max_auc > 0.75 and pheno_cur_auc < 0.73) and (ihm_max_auc > 0.85 and ihm_cur_auc < 0.83):
                self.model.stop_training = True


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
    a = softmax(a, axis=1, mask=mask)  # (B, T, 1)
    return K.sum(x * a, axis=1)  # (B, D)


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
            xt = tf.transpose(x, perm=(2, 0, 1))
            gt = tf.gather(xt, self.indices)
            return tf.transpose(gt, perm=(1, 2, 0))
        return x[:, :, self.indices]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.indices))

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def get_config(self):
        return {'indices': self.indices}


class GetTimestep(Layer):
    """ Takes 3D tensor and returns x[:, pos, :]
    """

    def __init__(self, pos=-1, **kwargs):
        self.pos = pos
        self.supports_masking = True
        super(GetTimestep, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, self.pos, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        return {'pos': self.pos}


LastTimestep = GetTimestep


class ExtendMask(Layer):
    """ Inputs:      [X, M]
        Output:      X
        Output_mask: M
    """

    def __init__(self, add_epsilon=False, **kwargs):
        self.supports_masking = True
        self.add_epsilon = add_epsilon
        super(ExtendMask, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, input, input_mask=None):
        if self.add_epsilon:
            return input[1] + K.epsilon()
        return input[1]

    def get_config(self):
        return {'add_epsilon': self.add_epsilon}
