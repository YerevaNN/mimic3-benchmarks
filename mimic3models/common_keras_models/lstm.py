from __future__ import absolute_import
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep


class Network(Model):
    
    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                target_repl, num_classes=1, depth=1, input_dim=76, **kwargs):
        
        print "==> not used params in network class:", kwargs.keys()
        
        # TODO: recurrent batch normalization
        
        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
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

        X = Input(shape=(None, input_dim), name='X')
        mX = Masking()(X)

        for i in range(depth - 1):
            mX = Bidirectional(LSTM(units=dim//2,
                                   activation='tanh',
                                   return_sequences=True,
                                   recurrent_dropout=rec_dropout,
                                   dropout=dropout))(mX)

        L = LSTM(units=dim,
                 activation='tanh',
                 return_sequences=(self.target_repl > 0),
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX)

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
        self.network_class_name = "k_lstm"
        return "{}.n{}{}{}{}.dep{}{}".format(self.network_class_name,
                    self.dim,
                    ".bn" if self.batch_norm else "",
                    ".d{}".format(self.dropout) if self.dropout > 0 else "",
                    ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                    self.depth,
                    ".trc{}".format(self.target_repl) if self.target_repl > 0 else "")
