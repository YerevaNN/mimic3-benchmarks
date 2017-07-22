from __future__ import absolute_import
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout, Reshape, Activation
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import CollectAttetion, Slice
from keras.layers.merge import Concatenate


class Network(Model):
    
    def __init__(self, dim, batch_norm, dropout, rec_dropout, batch_size,
                header, task, num_classes=1, depth=1,
                input_dim=76, **kwargs):
        
        # TODO: recurrent batch normalization
        
        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        
        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu' # NOTE: what if it is regression but in log-space
            else:
                final_activation = 'softmax'
        else:
            return ValueError("Wrong value for task")
        
        print "==> not used params in network class:", kwargs.keys()
        
        # Parse channels
        channel_names = set()
        for ch in header:
            if ch.find("mask->") != -1:
                continue
            pos = ch.find("->")
            if pos != -1:
                channel_names.add(ch[:pos])
            else:
                channel_names.add(ch)
        channel_names = sorted(list(channel_names))
        print "==> found {} channels: {}".format(len(channel_names), channel_names)
        
        channels = [] # each channel is a list of columns
        for ch in channel_names:
            indices = range(len(header))
            indices = filter(lambda i: header[i].find(ch) != -1, indices)
            channels.append(indices)
        
        X = Input(shape=(None, input_dim), name='X')
        mX = Masking()(X)
        
        cX = [] # channel X
        for ch in channels:
            cX.append(Slice(ch)(mX))
        
        pX = [] # LSTM processed version of cX
        for x in cX:
            p = x
            for i in range(depth):
                p = Bidirectional(LSTM(units=dim//2,
                                   activation='tanh',
                                   return_sequences=True,
                                   dropout=dropout,
                                   recurrent_dropout=rec_dropout))(p)
            pX.append(p)
        
        # Concatenate
        Z = Concatenate(axis=2)(pX)
        
        for i in range(depth-1):
            Z = Bidirectional(LSTM(units=4*dim//2,
                            activation='tanh',
                            return_sequences=True,
                            dropout=dropout,
                            recurrent_dropout=rec_dropout))(Z)
        
        L = LSTM(units=4*dim,
                 activation='tanh',
                 return_sequences=False,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(Z)
        
        if dropout > 0:
            L = Dropout(dropout)(L)
        
        y = Dense(num_classes, activation=final_activation)(L)
        
        return super(Network, self).__init__(inputs=[X],
                                             outputs=[y])
    
    
    def say_name(self):
        self.network_class_name = "k_channel_wise_lstms"
        return "{}.n{}{}{}{}.dep{}".format(self.network_class_name,
                    self.dim,
                    ".bn" if self.batch_norm else "",
                    ".d{}".format(self.dropout) if self.dropout > 0 else "",
                    ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                    self.depth)
