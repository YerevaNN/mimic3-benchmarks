import random
import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne.nonlinearities import softmax, sigmoid, tanh, rectify
from lasagne.objectives import categorical_crossentropy, squared_error
from lasagne.init import Orthogonal, Normal

from mimic3models import nn_utils
from mimic3models import metrics

floatX = theano.config.floatX


class Network(nn_utils.BaseNetwork):
    
    def __init__(self, train_raw, test_raw, dim, mode, l2, l1,
                 batch_norm, dropout, batch_size,
                 ihm_C, los_C, ph_C, decomp_C,
                 partition, nbins, **kwargs):
                
        print "==> not used params in network class:", kwargs.keys()
        self.train_raw = train_raw
        self.test_raw = test_raw
        
        self.dim = dim
        self.mode = mode
        self.l2 = l2
        self.l1 = l1
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.batch_size = batch_size
        self.ihm_C = ihm_C
        self.los_C = los_C
        self.ph_C = ph_C
        self.decomp_C = decomp_C
        self.nbins = nbins
        
        if (partition == 'log'):
            self.get_bin = metrics.get_bin_log
            self.get_estimate = metrics.get_estimate_log
        else:
            assert self.nbins == 10
            self.get_bin = metrics.get_bin_custom
            self.get_estimate = metrics.get_estimate_custom
        
        self.train_batch_gen = self.get_batch_gen(self.train_raw)
        self.test_batch_gen = self.get_batch_gen(self.test_raw)    
        
        self.input_var = T.tensor3('X')
        self.input_lens = T.ivector('L')
        
        self.ihm_pos = T.ivector('ihm_pos')
        self.ihm_mask = T.ivector('ihm_mask')
        self.ihm_label = T.ivector('ihm_label')
        
        self.los_mask = T.imatrix('los_mask')
        self.los_label = T.matrix('los_label') # for regression
        #self.los_label = T.imatrix('los_label')
        
        self.ph_label = T.imatrix('ph_label')
        
        self.decomp_mask = T.imatrix('decomp_mask')
        self.decomp_label = T.imatrix('decomp_label')
        
        print "==> Building neural network"
        
        # common network
        network = layers.InputLayer((None, None, self.train_raw[0][0].shape[1]), 
                                    input_var=self.input_var)
        
        if (self.dropout > 0):
            network = layers.DropoutLayer(network, p=self.dropout)
        
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
        
        if (self.dropout > 0):
            network = layers.DropoutLayer(network, p=self.dropout)
        
        lstm_output = layers.get_output(network)
        self.params = layers.get_all_params(network, trainable=True)
        self.reg_params = layers.get_all_params(network, regularizable=True)
        
        # for each example in minibatch take the last output
        last_outputs = []
        for index in range(self.batch_size):
            last_outputs.append(lstm_output[index, self.input_lens[index]-1, :])
        last_outputs = T.stack(last_outputs)
        
        # take 48h outputs for fixed mortality task
        mid_outputs = []
        for index in range(self.batch_size):
            mid_outputs.append(lstm_output[index, self.ihm_pos[index], :])
        mid_outputs = T.stack(mid_outputs)
        
        
        # in-hospital mortality related network
        ihm_network = layers.InputLayer((None, dim), input_var=mid_outputs)
        ihm_network = layers.DenseLayer(incoming=ihm_network, num_units=2,
                                       nonlinearity=softmax)
        self.ihm_prediction = layers.get_output(ihm_network)
        self.ihm_det_prediction = layers.get_output(ihm_network, deterministic=True)
        self.params += layers.get_all_params(ihm_network, trainable=True)
        self.reg_params += layers.get_all_params(ihm_network, regularizable=True)
        self.ihm_loss = (self.ihm_mask * categorical_crossentropy(self.ihm_prediction, 
                                                          self.ihm_label)).mean()
        
        
        # length of stay related network
        # Regression
        los_network = layers.InputLayer((None, None, dim), input_var=lstm_output)
        los_network = layers.ReshapeLayer(los_network, (-1, dim))
        los_network = layers.DenseLayer(incoming=los_network, num_units=1,
                                        nonlinearity=rectify)
        los_network = layers.ReshapeLayer(los_network, (lstm_output.shape[0], -1))
        self.los_prediction = layers.get_output(los_network)
        self.los_det_prediction = layers.get_output(los_network, deterministic=True)
        self.params += layers.get_all_params(los_network, trainable=True)
        self.reg_params += layers.get_all_params(los_network, regularizable=True)
        self.los_loss = (self.los_mask * squared_error(self.los_prediction,
                                                      self.los_label)).mean(axis=1).mean(axis=0)
        
        
        # phenotype related network
        ph_network = layers.InputLayer((None, dim), input_var=last_outputs)
        ph_network = layers.DenseLayer(incoming=ph_network, num_units=25,
                                       nonlinearity=sigmoid)
        self.ph_prediction = layers.get_output(ph_network)
        self.ph_det_prediction = layers.get_output(ph_network, deterministic=True)
        self.params += layers.get_all_params(ph_network, trainable=True)
        self.reg_params += layers.get_all_params(ph_network, regularizable=True)
        self.ph_loss = nn_utils.multilabel_loss(self.ph_prediction, self.ph_label)
                
        
        # decompensation related network
        decomp_network = layers.InputLayer((None, None, dim), input_var=lstm_output)
        decomp_network = layers.ReshapeLayer(decomp_network, (-1, dim))
        decomp_network = layers.DenseLayer(incoming=decomp_network, num_units=2,
                                       nonlinearity=softmax)
        decomp_network = layers.ReshapeLayer(decomp_network, (lstm_output.shape[0], -1, 2))
        self.decomp_prediction = layers.get_output(decomp_network)[:, :, 1]
        self.decomp_det_prediction = layers.get_output(decomp_network, deterministic=True)[:, :, 1]
        self.params += layers.get_all_params(decomp_network, trainable=True)
        self.reg_params += layers.get_all_params(decomp_network, regularizable=True)
        self.decomp_loss = nn_utils.multilabel_loss_with_mask(self.decomp_prediction,
                                                          self.decomp_label,
                                                          self.decomp_mask)
        
        """
        data = next(self.train_batch_gen)
        print max(data[1])
        print lstm_output.eval({self.input_var:data[0]}).shape
        exit()
        """
        
        
        if self.l2 > 0: 
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.reg_params)
        else: 
            self.loss_l2 = T.constant(0)
        
        if self.l1 > 0: 
            self.loss_l1 = self.l1 * nn_utils.l1_reg(self.reg_params)
        else: 
            self.loss_l1 = T.constant(0)
        
        self.reg_loss = self.loss_l1 + self.loss_l2
        
        self.loss = (ihm_C * self.ihm_loss + los_C * self.los_loss + 
                     ph_C * self.ph_loss + decomp_C * self.decomp_loss + 
                     self.reg_loss)
              
        #updates = lasagne.updates.adadelta(self.loss, self.params,
        #                                    learning_rate=0.001)
        #updates = lasagne.updates.momentum(self.loss, self.params,
        #                                    learning_rate=0.00003)
        #updates = lasagne.updates.adam(self.loss, self.params)
        updates = lasagne.updates.adam(self.loss, self.params, beta1=0.5,
                                       learning_rate=0.0001) # from DCGAN paper
        #updates = lasagne.updates.nesterov_momentum(loss, params, momentum=0.9,
        #                                             learning_rate=0.001,
        
        all_inputs = [self.input_var, self.input_lens,
                      self.ihm_pos, self.ihm_mask, self.ihm_label,
                      self.los_mask, self.los_label,
                      self.ph_label,
                      self.decomp_mask, self.decomp_label]
        
        train_outputs = [self.ihm_prediction, self.los_prediction,
                         self.ph_prediction, self.decomp_prediction,
                         self.loss,
                         self.ihm_loss, self.los_loss,
                         self.ph_loss, self.decomp_loss,
                         self.reg_loss]
                         
        test_outputs = [self.ihm_det_prediction, self.los_det_prediction,
                        self.ph_det_prediction, self.decomp_det_prediction,
                        self.loss,
                        self.ihm_loss, self.los_loss,
                        self.ph_loss, self.decomp_loss,
                        self.reg_loss]
        
        ## compiling theano functions
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=all_inputs,
                                            outputs=train_outputs,
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=all_inputs,
                                       outputs=test_outputs)
        
        
    def process_input(self, data_raw):
        X = nn_utils.pad_zeros(data_raw[0]).astype(np.float32)
        lens = np.array(map(len, data_raw[0]), dtype=np.int32)
        
        fms = data_raw[1]
        loss = data_raw[2]
        phs = data_raw[3]
        sws = data_raw[4]
        
        ihm_pos = np.array([x[0] for x in fms], dtype=np.int32)
        ihm_mask = np.array([x[1] for x in fms], dtype=np.int32)
        ihm_label = np.array([x[2] for x in fms], dtype=np.int32)
        
        los_mask = [np.array(x[0], dtype=np.int32) for x in loss]
        los_mask = nn_utils.pad_zeros(los_mask).astype(np.int32)
        
        los_label = [np.array(x[1], dtype=np.float32) for x in loss]
        los_label = np.log(1.0 + nn_utils.pad_zeros(los_label)).astype(np.float32)
        
        ph_label = [np.array(x, dtype=np.int32) for x in phs]
        ph_label = nn_utils.pad_zeros(ph_label).astype(np.int32)
        
        decomp_mask = [np.array(x[0], dtype=np.int32) for x in sws]
        decomp_mask = nn_utils.pad_zeros(decomp_mask).astype(np.int32)
        
        decomp_label = [np.array(x[1], dtype=np.int32) for x in sws]
        decomp_label = nn_utils.pad_zeros(decomp_label).astype(np.int32)
        
        return (X, lens,
                ihm_pos, ihm_mask, ihm_label,
                los_mask, los_label,
                ph_label,
                decomp_mask, decomp_label)
    
    
    def say_name(self):
        self.network_class_name = "lstm_logspace"
        network_name = '%s.n%d%s%s.bs%d%s%s.%.2f.%.2f.%.2f.%.2f' % (self.network_class_name,
                        self.dim, (".bn" if self.batch_norm else ""), 
                        (".d" + str(self.dropout)) if self.dropout > 0 else "",
                        self.batch_size,
                        ".L2%f" % self.l2 if self.l2 > 0 else "",
                        ".L1%f" % self.l1 if self.l1 > 0 else "",
                        self.ihm_C, self.los_C, self.ph_C, self.decomp_C)
        return network_name
    
    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train' or mode == 'predict_on_train'):
            return len(self.train_raw[0]) // self.batch_size
        elif (mode == 'test' or mode == 'predict'):
            return len(self.test_raw[0]) // self.batch_size
        else:
            raise Exception("unknown mode")
    
    
    def shuffle_and_sort(self, data):
        assert(len(data) == 5)
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
        # so all examples in one batch will have more or less the same lenghts
        
        assert len(data) == old_size
        data = zip(*data)
        assert(len(data) == 5)
        assert(len(data[0]) == old_size)
        assert(len(data[1]) == old_size)
        return data
    
    
    def shuffle_train_set(self):
        print "==> Shuffling the train set"
        self.train_raw = self.shuffle_and_sort(self.train_raw)
        self.test_raw = self.shuffle_and_sort(self.test_raw)
        self.train_batch_gen = self.get_batch_gen(self.train_raw)
        self.test_batch_gen = self.get_batch_gen(self.test_raw) 


    def get_batch_gen(self, data):
        index = 0
        n = len(data[0])
        while True:
            if (index + self.batch_size > n):
                index = 0
            ret = (data[0][index:index + self.batch_size],
                   data[1][index:index + self.batch_size],
                   data[2][index:index + self.batch_size],
                   data[3][index:index + self.batch_size],
                   data[4][index:index + self.batch_size])
            index += self.batch_size
            yield self.process_input(ret), ret
    
    
    def get_estimates(self, predictions):
        return np.array([[self.get_estimate(p, self.nbins) for p in x] for x in predictions],
                        dtype=np.int32)
            
    
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
            
        data, orig_data = next(batch_gen)
        ret = theano_fn(*data)
        
        return {"ihm_prediction": ret[0],
                "los_prediction": np.exp(ret[1])-1,
                "ph_prediction": ret[2],
                "decomp_prediction": ret[3],
                "loss": ret[4],
                "ihm_loss": ret[5],
                "los_loss": ret[6],
                "ph_loss": ret[7],
                "decomp_loss": ret[8],
                "reg_loss": ret[9],
                "log": "",
                "data": orig_data}
                
    
    def predict(self, data):
        """ data is a pair (X, y) """
        processed = self.process_input(data)
        ret = self.test_fn(*processed)
        return {"ihm_prediction": ret[0],
                "los_prediction": np.exp(ret[1])-1,
                "ph_prediction": ret[2],
                "decomp_prediction": ret[3],
                "loss": ret[4],
                "ihm_loss": ret[5],
                "los_loss": ret[6],
                "ph_loss": ret[7],
                "decomp_loss": ret[8],
                "reg_loss": ret[9]}
