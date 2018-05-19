from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import Slice, LastTimestep
from keras.layers.merge import Concatenate
from mimic3models.keras_utils import ExtendMask


class Network(Model):

    def __init__(self, dim, batch_norm, dropout, rec_dropout, header, task,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=76, size_coef=4, **kwargs):

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.size_coef = size_coef

        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")

        print("==> not used params in network class:", kwargs.keys())

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
        print("==> found {} channels: {}".format(len(channel_names), channel_names))

        channels = []  # each channel is a list of columns
        for ch in channel_names:
            indices = range(len(header))
            indices = list(filter(lambda i: header[i].find(ch) != -1, indices))
            channels.append(indices)

        # Input layers and masking
        X = Input(shape=(None, input_dim), name='X')
        inputs = [X]
        mX = Masking()(X)

        if deep_supervision:
            M = Input(shape=(None,), name='M')
            inputs.append(M)

        # Configurations
        is_bidirectional = True
        if deep_supervision:
            is_bidirectional = False

        # Preprocess each channel
        cX = []
        for ch in channels:
            cX.append(Slice(ch)(mX))
        pX = []  # LSTM processed version of cX
        for x in cX:
            p = x
            for i in range(depth):
                num_units = dim
                if is_bidirectional:
                    num_units = num_units // 2

                lstm = LSTM(units=num_units,
                            activation='tanh',
                            return_sequences=True,
                            dropout=dropout,
                            recurrent_dropout=rec_dropout)

                if is_bidirectional:
                    p = Bidirectional(lstm)(p)
                else:
                    p = lstm(p)
            pX.append(p)

        # Concatenate processed channels
        Z = Concatenate(axis=2)(pX)

        # Main part of the network
        for i in range(depth-1):
            num_units = int(size_coef*dim)
            if is_bidirectional:
                num_units = num_units // 2

            lstm = LSTM(units=num_units,
                        activation='tanh',
                        return_sequences=True,
                        dropout=dropout,
                        recurrent_dropout=rec_dropout)

            if is_bidirectional:
                Z = Bidirectional(lstm)(Z)
            else:
                Z = lstm(Z)

        # Output module of the network
        return_sequences = (target_repl or deep_supervision)
        L = LSTM(units=int(size_coef*dim),
                 activation='tanh',
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(Z)

        if dropout > 0:
            L = Dropout(dropout)(L)

        if target_repl:
            y = TimeDistributed(Dense(num_classes, activation=final_activation),
                                name='seq')(L)
            y_last = LastTimestep(name='single')(y)
            outputs = [y_last, y]
        elif deep_supervision:
            y = TimeDistributed(Dense(num_classes, activation=final_activation))(L)
            y = ExtendMask()([y, M])  # this way we extend mask of y to M
            outputs = [y]
        else:
            y = Dense(num_classes, activation=final_activation)(L)
            outputs = [y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}.szc{}{}{}{}.dep{}".format('k_channel_wise_lstms',
                                                 self.dim,
                                                 self.size_coef,
                                                 ".bn" if self.batch_norm else "",
                                                 ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                                 ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                                 self.depth)
