import numpy as np
import metrics

import keras
import keras.backend as K
from keras.layers import Layer


class MetricsBinaryFromGenerator(keras.callbacks.Callback):
    
    def __init__(self, train_data_gen, val_data_gen, batch_size=32):
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
    
    
    def on_train_begin(self, logs={}):
        self.train_y_true = []
        self.train_predictions = []
        self.train_auroc = []
        self.train_auprc = []
        self.train_minpse = []
        
        self.val_y_true = []
        self.val_preictions = []
        self.val_auroc = []
        self.val_auprc = []
        self.val_minpse = []
    
    
    def calc_metrics(self, data_gen, auroc, auprc, minpse):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            (x,y) = next(data_gen)
            y_true += list(y)
            predictions += list(self.model.predict(x, batch_size=self.batch_size))
        
        predictions = np.array(predictions)
        predictions = np.stack([1-predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        auroc.append(ret['auroc'])
        auprc.append(ret['auprc'])
        minpse.append(ret['minpse'])
    
    
    def on_epoch_end(self, epoch, logs={}):
        
        print "\t predicting on train"
        self.calc_metrics(self.train_data_gen,
                        self.train_auroc,
                        self.train_auprc,
                        self.train_minpse)
        
        print "\t predicting on validation"
        self.calc_metrics(self.val_data_gen,
                        self.val_auroc,
                        self.val_auprc,
                        self.val_minpse)


class MetricsBinaryFromData(keras.callbacks.Callback):
    
    def __init__(self, train_data, val_data, batch_size=32):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
    
    
    def on_train_begin(self, logs={}):
        self.train_y_true = []
        self.train_predictions = []
        self.train_auroc = []
        self.train_auprc = []
        self.train_minpse = []
        
        self.val_y_true = []
        self.val_preictions = []
        self.val_auroc = []
        self.val_auprc = []
        self.val_minpse = []
    
    
    def calc_metrics(self, data, auroc, auprc, minpse):
        y_true = []
        predictions = []
        num_examples = len(data[0])
        for i in range(0, num_examples, self.batch_size):
            (x,y) = (data[0][i:i+self.batch_size], data[1][i:i+self.batch_size])
            y_true += list(y)
            predictions += list(self.model.predict(x, batch_size=self.batch_size))
        
        predictions = np.array(predictions)
        predictions = np.stack([1-predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        auroc.append(ret['auroc'])
        auprc.append(ret['auprc'])
        minpse.append(ret['minpse'])
    
    
    def on_epoch_end(self, epoch, logs={}):
        
        print "\t predicting on train"
        self.calc_metrics(self.train_data,
                        self.train_auroc,
                        self.train_auprc,
                        self.train_minpse)
        
        print "\t predicting on validation"
        self.calc_metrics(self.val_data,
                        self.val_auroc,
                        self.val_auprc,
                        self.val_minpse)
                        
                        
class CollectAttetion(Layer):
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
