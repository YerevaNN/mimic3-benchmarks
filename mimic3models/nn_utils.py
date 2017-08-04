import os
import theano
import theano.tensor as T
import lasagne
import cPickle as pickle
import numpy as np


class BaseNetwork:
    
    def say_name(self):
        return "unknown"
    
    
    def save_params(self, file_name, epoch, **kwargs):
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1)
    
    
    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        epoch = 0
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            assert(len(self.params) == len(loaded_params))
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)
            epoch = dict['epoch']
        return epoch


def softmax(x):
    e_x = T.exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)


def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])
    

def l1_reg(params):
    return T.sum([T.sum(abs(x)) for x in params])


def constant_param(value=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Constant(value).sample(shape), borrow=True)
    

def normal_param(std=0.1, mean=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Normal(std, mean).sample(shape), borrow=True)


def multilabel_loss(preds, labels):
    eps = 1e-4
    preds = T.clip(preds, eps, 1-eps)
    return -(labels * T.log(preds) + (1 - labels) * T.log(1 - preds)).mean(axis=1).mean(axis=0)


def multilabel_loss_with_mask(preds, labels, mask):
    eps = 1e-4
    preds = T.clip(preds, eps, 1-eps)
    return -(mask * (labels * T.log(preds) + (1 - labels) * T.log(1 - preds))).mean(axis=1).mean(axis=0)


def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s
    
    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) 
                for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
                for x in ret]
    return np.array(ret)
    

def pad_zeros_from_left(arr):
    """
    `arr` is an array of `np.array`s
    
    The function appends zeros from left to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([np.zeros((max_len - x.shape[0],) + x.shape[1:]), x], axis=0) 
                for x in arr]
    return np.array(ret)
