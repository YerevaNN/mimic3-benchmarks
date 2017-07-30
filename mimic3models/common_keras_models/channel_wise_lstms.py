from __future__ import absolute_import
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import Slice, LastTimestep
from keras.layers.merge import Concatenate


class Network(Model):

    def __init__(self, dim, batch_norm, dropout, rec_dropout, batch_size,
                header, task, target_repl, num_classes=1, depth=1,
                input_dim=76, size_coef=4, **kwargs):

        # TODO: recurrent batch normalization

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.size_coef = size_coef
        self.target_repl = target_repl

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
            Z = Bidirectional(LSTM(units=int(size_coef*dim)//2,
                            activation='tanh',
                            return_sequences=True,
                            dropout=dropout,
                            recurrent_dropout=rec_dropout))(Z)

        L = LSTM(units=int(size_coef*dim),
                 activation='tanh',
                 return_sequences=(self.target_repl > 0),
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(Z)

        if dropout > 0:
            L = Dropout(dropout)(L)

        if self.target_repl > 0:
            y = TimeDistributed(Dense(num_classes, activation=final_activation),
                                name='seq')(L)
            y_last = LastTimestep(name='single')(y)
            outputs = [y_last, y]
        else:
            y = Dense(num_classes, activation=final_activation, name='single')(L)
            outputs = [y]

        return super(Network, self).__init__(inputs=[X],
                                             outputs=outputs)
    
    def say_name(self):
        self.network_class_name = "k_channel_wise_lstms"
        return "{}.n{}.szc{}{}{}{}.dep{}{}".format(self.network_class_name,
                    self.dim,
                    self.size_coef,
                    ".bn" if self.batch_norm else "",
                    ".d{}".format(self.dropout) if self.dropout > 0 else "",
                    ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                    self.depth,
                    ".trc{}".format(self.target_repl) if self.target_repl > 0 else "")
