from __future__ import absolute_import
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import Bidirectional

class Network(Model):
    
    def __init__(self, dim, batch_norm, dropout, rec_dropout, task, num_classes=1,
                depth=1, input_dim=76, **kwargs):
        
        print "==> not used params in network class:", kwargs.keys()
        
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
                 return_sequences=False,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX)
        
        if dropout > 0:
            L = Dropout(dropout)(L)

        y = Dense(num_classes, activation=final_activation)(L)

        return super(Network, self).__init__(inputs=[X],
                                             outputs=[y])


    def say_name(self):
        self.network_class_name = "k_lstm"
        return "{}.n{}{}{}{}.dep{}".format(self.network_class_name,
                    self.dim,
                    ".bn" if self.batch_norm else "",
                    ".d{}".format(self.dropout) if self.dropout > 0 else "",
                    ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                    self.depth)
