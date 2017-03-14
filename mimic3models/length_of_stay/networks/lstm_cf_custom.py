import random
import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne.nonlinearities import softmax, rectify
from lasagne.objectives import squared_error, categorical_crossentropy
from lasagne.init import Orthogonal, Normal

from mimic3models import nn_utils
from mimic3models import metrics

floatX = theano.config.floatX


class Network(nn_utils.BaseNetwork):
    
    def __init__(self, dim, mode, l2, l1, batch_norm, dropout,
                 batch_size, input_dim=76, **kwargs):
                
        print "==> not used params in network class:", kwargs.keys()
        
        self.nbins = 10
        
        self.dim = dim
        self.mode = mode
        self.l2 = l2
        self.l1 = l1
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.batch_size = batch_size
        
        self.input_var = T.tensor3('X')
        self.input_lens = T.ivector('L')
        self.target_var = T.ivector('y')
        
        print "==> Building neural network"
        network = layers.InputLayer((None, None, input_dim), 
                                    input_var=self.input_var)
        network = layers.LSTMLayer(incoming=network, num_units=dim,
                                   only_return_final=False,
                                   grad_clipping=10,
                                   ingate=lasagne.layers.Gate(
                                        W_in=Orthogonal(),
                                        W_hid=Orthogonal(),
                                        W_cell=Normal(0.1)),
                                   forgetgate=lasagne.layers.Gate(
                                        W_in=Orthogonal(),
                                        W_hid=Orthogonal(),
                                        W_cell=Normal(0.1)),
                                   cell=lasagne.layers.Gate(W_cell=None,
                                        nonlinearity=lasagne.nonlinearities.tanh,
                                        W_in=Orthogonal(),
                                        W_hid=Orthogonal()),
                                   outgate=lasagne.layers.Gate(
                                        W_in=Orthogonal(),
                                        W_hid=Orthogonal(),
                                        W_cell=Normal(0.1)))
        lstm_output = layers.get_output(network)
        self.params = layers.get_all_params(network, trainable=True)
        self.reg_params = layers.get_all_params(network, regularizable=True)
        
        # for each example in minibatch take the last output
        last_outputs = []
        for index in range(self.batch_size):
            last_outputs.append(lstm_output[index, self.input_lens[index]-1, :])
        last_outputs = T.stack(last_outputs)

        network = layers.InputLayer(shape=(self.batch_size, self.dim), 
                                    input_var=last_outputs)
        network = layers.DenseLayer(incoming=network, num_units=self.nbins,
                                    nonlinearity=softmax)
        
        self.prediction = layers.get_output(network)
        self.params += layers.get_all_params(network, trainable=True)
        self.reg_params += layers.get_all_params(network, regularizable=True)
        
        self.loss_ce = categorical_crossentropy(self.prediction,
                                                self.target_var).mean()    
        
        if self.l2 > 0: 
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.reg_params)
        else: 
            self.loss_l2 = T.constant(0)
        
        if self.l1 > 0: 
            self.loss_l1 = self.l1 * nn_utils.l1_reg(self.reg_params)
        else: 
            self.loss_l1 = T.constant(0)
            
        self.loss_reg = self.loss_l1 + self.loss_l2
            
        self.loss = self.loss_ce + self.loss_reg
        
        #updates = lasagne.updates.adadelta(self.loss, self.params,
        #                                    learning_rate=0.001)
        #updates = lasagne.updates.momentum(self.loss, self.params,
        #                                    learning_rate=0.00003)
        #updates = lasagne.updates.adam(self.loss, self.params)
        updates = lasagne.updates.adam(self.loss, self.params, beta1=0.5,
                                       learning_rate=0.0001) # from DCGAN paper
        #updates = lasagne.updates.nesterov_momentum(loss, params, momentum=0.9,
        #                                             learning_rate=0.001,
        
        ## compiling theano functions
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.input_var,
                                                    self.input_lens,
                                                    self.target_var],
                                            outputs=[self.prediction, self.loss, self.loss_reg],
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.input_var,
                                               self.input_lens,
                                               self.target_var],
                                       outputs=[self.prediction, self.loss, self.loss_reg])
    
    def set_datasets(self, train_raw, test_raw):
        self.train_raw = train_raw
        self.test_raw = test_raw
        self.train_batch_gen = self.get_batch_gen(self.train_raw)
        self.test_batch_gen = self.get_batch_gen(self.test_raw)
        
        
    def process_input(self, data_raw):
        Xs = nn_utils.pad_zeros(data_raw[0]).astype(np.float32)
        lens = map(len, data_raw[0])
        ys = np.array(data_raw[1]).astype(np.float32)
        
        bin_ids = [metrics.get_bin_custom(x, self.nbins) for x in ys]

        for x in bin_ids:
            assert x >= 0 and x < self.nbins
        
        return (Xs, lens, np.array(bin_ids, dtype=np.int32), ys)
    
    
    def say_name(self):
        self.network_class_name = "lstm_cf_custom"
        network_name = '%s.n%d%s%s.bs%d%s%s' % (self.network_class_name,
                        self.dim, (".bn" if self.batch_norm else ""), 
                        (".d" + str(self.dropout)) if self.dropout > 0 else "",
                        self.batch_size,
                        ".L2%f" % self.l2 if self.l2 > 0 else "",
                        ".L1%f" % self.l1 if self.l1 > 0 else "")
        return network_name
    

    def get_batches_per_epoch(self, mode):
        if (mode == 'train' or mode == 'predict_on_train'):
            return len(self.train_raw[0]) // self.batch_size
        elif (mode == 'test' or mode == 'predict'):
            return len(self.test_raw[0]) // self.batch_size
        else:
            raise Exception("unknown mode")

    
    def shuffle_and_sort(self, data):
        assert(len(data) == 2)
        data = zip(*data)
        random.shuffle(data)
        
        old_size = len(data)
        rem = old_size % self.batch_size
        head = data[:old_size - rem]
        tail = data[old_size - rem:]
        data = []
        
        head.sort(key=(lambda x: x[0].shape[0]))
        
        size = len(head)
        mas = [head[i : i+self.batch_size] for i in range(0, size, self.batch_size)]
        random.shuffle(mas)
        
        for x in mas:
            data += x
        data += tail
        # NOTE: we assume that we will not use cycling in batch generator
        # so all examples in one batch will have more or less the same context lenghts
        
        assert len(data) == old_size
        data = zip(*data)
        assert(len(data) == 2)
        assert(len(data[0]) == old_size)
        assert(len(data[1]) == old_size)
        return data
    
    
    def shuffle_train_set(self):
        print "==> Shuffling the train set"
        if (self.train_raw is not None):
            self.train_raw = self.shuffle_and_sort(self.train_raw)
            self.train_batch_gen = self.get_batch_gen(self.train_raw)
        if (self.test_raw is not None):
            self.test_raw = self.shuffle_and_sort(self.test_raw)
            self.test_batch_gen = self.get_batch_gen(self.test_raw) 
    

    def get_batch_gen(self, data):
        index = 0
        n = len(data[0])
        while True:
            if (index + self.batch_size > n):
                index = 0
            ret = (data[0][index:index + self.batch_size],
                   data[1][index:index + self.batch_size])
            index += self.batch_size
            yield self.process_input(ret)
    
    
    def step(self, mode):
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")

        if mode == "train":
            theano_fn = self.train_fn
            batch_gen = self.train_batch_gen
        elif mode == "test":    
            theano_fn = self.test_fn
            batch_gen = self.test_batch_gen
        else:
            raise Exception("Invalid mode")
            
        data = next(batch_gen)
        ys = data[-1]
        data = data[:-1]
        ret = theano_fn(*data)
        
        return {"prediction": [metrics.get_estimate_custom(x, self.nbins) for x in ret[0]],
                "answers": ys,
                "current_loss": ret[1],
                "loss_reg": ret[2],
                "loss_mse": ret[1] - ret[2],
                "log": ""}
                
    
    def predict(self, data):
        """ data is a pair (X, y) """
        processed = self.process_input(data)[:-1]
        ret = self.test_fn(*processed)
        predictions = [metrics.get_estimate_custom(x, self.nbins) for x in ret[0]]
        return [predictions] + list(ret[1:])