import numpy as np
from mimic3models import metrics
from mimic3models import nn_utils
from mimic3models import common_utils
import threading


def read_chunk(reader, chunk_size):
    data = []
    ts = []
    fms = []
    loss = []
    phs = []
    sws = []
    for i in range(chunk_size):
        (X, t, fm, los, ph, sw, header) = reader.read_next()
        data.append(X)
        ts.append(t)
        fms.append(fm)
        loss.append(los)
        phs.append(ph)
        sws.append(sw)
    return (data, ts, fms, loss, phs, sws)


def load_data(reader, discretizer, normalizer, small_part=False):
    N = reader.get_number_of_examples()
    if (small_part == True):
        N = 1000
    (data, ts, fms, loss, phs, sws) = read_chunk(reader, N)
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]
    return (data, fms, loss, phs, sws)


class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer, ihm_pos, partition,
                 target_repl, batch_size, small_part):

        N = reader.get_number_of_examples()
        if small_part:
            N = 1000

        self.discretizer = discretizer
        self.normalizer = normalizer
        self.ihm_pos = ihm_pos
        self.partition = partition
        self.target_repl = target_repl
        self.batch_size = batch_size
        self.steps = N // batch_size
        self.lock = threading.Lock()

        (Xs, ts, ihms, loss, phenos, decomps) = read_chunk(reader, N)

        for i in range(N):
            (Xs[i], ihms[i], decomps[i], loss[i], phenos[i]) = \
                self._preprocess_single(Xs[i], ts[i], ihms[i], decomps[i], loss[i], phenos[i])

        self.data = {}
        self.data['X'] = Xs
        self.data['ihm_M'] = [x[0] for x in ihms]
        self.data['ihm_y'] = [x[1] for x in ihms]
        self.data['decomp_M'] = [x[0] for x in decomps]
        self.data['decomp_y'] = [x[1] for x in decomps]
        self.data['los_M'] = [x[0] for x in loss]
        self.data['los_y'] = [x[1] for x in loss]
        self.data['pheno_y'] = phenos

        self.generator = self._generator()

    def _preprocess_single(self, X, max_time, ihm, decomp, los, pheno):
        timestep = self.discretizer._timestep
        eps = 1e-6

        def get_bin(t):
            return int(t / timestep - eps)

        sample_times = np.arange(0.0, max_time - eps, 1.0)
        sample_times = np.array([int(x+eps) for x in sample_times])
        assert len(sample_times) == len(decomp[0])
        assert len(sample_times) == len(los[0])

        nsteps = get_bin(max_time) + 1

        ## X
        X = self.discretizer.transform(X, end=max_time)[0]
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        assert len(X) == nsteps

        ## ihm
        # NOTE: when mask is 0, we set y to be 0. This is important
        #   because in the multitask networks when ihm_M = 0 we set
        #   our prediction thus the loss will be 0.
        if np.equal(ihm[1], 0):
            ihm[2] = 0
        ihm = (np.int32(ihm[1]), np.int32(ihm[2])) # mask, label

        ## decomp
        decomp_M = [0] * nsteps
        decomp_y = [0] * nsteps
        for (t, m, y) in zip(sample_times, decomp[0], decomp[1]):
            pos = get_bin(t)
            decomp_M[pos] = m
            decomp_y[pos] = y
        decomp = (np.array(decomp_M, dtype=np.int32),
                  np.array(decomp_y, dtype=np.int32))

        ## los
        los_M = [0] * nsteps
        los_y = [0] * nsteps
        for (t, m, y) in zip(sample_times, los[0], los[1]):
            pos = get_bin(t)
            los_M[pos] = m
            los_y[pos] = y
        los = (np.array(los_M, dtype=np.int32),
               np.array(los_y, dtype=np.float32))

        ## pheno
        pheno = np.array(pheno, dtype=np.int32)

        return (X, ihm, decomp, los, pheno)

    def _generator(self):
        B = self.batch_size
        while True:
            # convert to right format for sort_and_shuffle
            kvpairs = self.data.items()
            mas = [kv[1] for kv in kvpairs]
            mas = common_utils.sort_and_shuffle(mas, B)
            for i in range(len(kvpairs)):
                self.data[kvpairs[i][0]] = mas[i]

            for i in range(0, len(self.data['X']), B):
                outputs = []

                # X
                X = self.data['X'][i:i+B]
                X = nn_utils.pad_zeros(X, min_length=self.ihm_pos+1)
                T = X.shape[1]

                ## ihm
                ihm_M = np.array(self.data['ihm_M'][i:i+B])
                ihm_M = np.expand_dims(ihm_M, axis=-1) # (B, 1)
                ihm_y = np.array(self.data['ihm_y'][i:i+B])
                ihm_y = np.expand_dims(ihm_y, axis=-1) # (B, 1)
                outputs.append(ihm_y)
                if self.target_repl:
                    ihm_seq = np.expand_dims(ihm_y, axis=-1).repeat(T, axis=1) # (B, T, 1)
                    outputs.append(ihm_seq)

                ## decomp
                decomp_M = self.data['decomp_M'][i:i+B]
                decomp_M = nn_utils.pad_zeros(decomp_M, min_length=self.ihm_pos+1)
                decomp_y = self.data['decomp_y'][i:i+B]
                decomp_y = nn_utils.pad_zeros(decomp_y, min_length=self.ihm_pos+1)
                decomp_y = np.expand_dims(decomp_y, axis=-1) # (B, T, 1)
                outputs.append(decomp_y)

                ## los
                los_M = self.data['los_M'][i:i+B]
                los_M = nn_utils.pad_zeros(los_M, min_length=self.ihm_pos+1)
                los_y = self.data['los_y'][i:i+B]
                los_y_true = nn_utils.pad_zeros(los_y, min_length=self.ihm_pos+1)

                if self.partition == 'log':
                    los_y = [np.array([metrics.get_bin_log(x, 10) for x in z]) for z in los_y]
                if self.partition == 'custom':
                    los_y = [np.array([metrics.get_bin_custom(x, 10) for x in z]) for z in los_y]
                los_y = nn_utils.pad_zeros(los_y, min_length=self.ihm_pos+1)
                los_y = np.expand_dims(los_y, axis=-1) # (B, T, 1)
                outputs.append(los_y)

                ## pheno
                pheno_y = np.array(self.data['pheno_y'][i:i+B])
                outputs.append(pheno_y)
                if self.target_repl:
                    pheno_seq = np.expand_dims(pheno_y, axis=1).repeat(T, axis=1) # (B, T, 25)
                    outputs.append(pheno_seq)

                inputs = [X, ihm_M, decomp_M, los_M]

                if self.return_y_true:
                    yield (inputs, outputs, los_y_true)
                else:
                    yield (inputs, outputs)

    def __iter__(self):
        return self.generator

    def next(self, return_y_true=False):
        with self.lock:
            self.return_y_true = return_y_true
            return self.generator.next()

    def __next__(self):
        return self.generator.__next__()
