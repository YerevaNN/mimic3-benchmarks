from mimic3models import metrics
from mimic3models import common_utils
from mimic3models import nn_utils
import threading
import os
import numpy as np
import random


def read_chunk(reader, chunk_size):
    data = []
    ys = []
    ts = []
    header = None
    for i in range(chunk_size):
        (X, t, y, header) = reader.read_next()
        data.append(X)
        ts.append(t)
        ys.append(y)
    return (data, ts, ys, header)


def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]
    return data


class BatchGen(object):

    def __init__(self, reader, partition, discretizer, normalizer,
                 batch_size, steps, shuffle):
        self.reader = reader
        self.partition = partition
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size

        if steps is None:
            self.n_examples = reader.get_number_of_examples()
            self.steps = (self.n_examples + batch_size - 1) // batch_size
        else:
            self.n_examples = steps * batch_size
            self.steps = steps

        self.shuffle = shuffle
        self.chunk_size = min(1024, steps) * batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                remaining -= current_size
                (data, ts, labels, header) = read_chunk(self.reader, current_size)
                data = preprocess_chunk(data, ts, self.discretizer, self.normalizer)
                data = (data, labels)
                data = common_utils.sort_and_shuffle(data, B)

                for i in range(0, current_size, B):
                    X = nn_utils.pad_zeros(data[0][i:i+B])
                    y = data[1][i:i+B]
                    y_true = np.array(y)

                    if self.partition == 'log':
                        y = [metrics.get_bin_log(x, 10) for x in y]
                    if self.partition == 'custom':
                        y = [metrics.get_bin_custom(x, 10) for x in y]

                    y = np.array(y)

                    if self.return_y_true:
                        yield (X, y, y_true)
                    else:
                        yield (X, y)

    def __iter__(self):
        return self.generator

    def next(self, return_y_true=False):
        with self.lock:
            self.return_y_true = return_y_true
            return self.generator.next()

    def __next__(self):
        return self.generator.__next__()


class BatchGenDeepSupervisoin(object):

    def __init__(self, dataloader, partition, discretizer, normalizer, batch_size, shuffle):
        self.partition = partition
        self.batch_size = batch_size
        self.data = self._load_per_patient_data(dataloader, discretizer, normalizer)
        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.shuffle = shuffle
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_per_patient_data(self, dataloader, discretizer, normalizer):
        timestep = discretizer._timestep

        def get_bin(t):
            eps = 1e-6
            return int(t / timestep - eps)

        N = len(dataloader._data)
        Xs = []
        masks = []
        ys = []

        for i in range(N):
            (X, positions, labels) = dataloader._data[i]
            labels = [float(x) for x in labels]

            T = max(positions)
            nsteps = get_bin(T) + 1
            mask = [0] * nsteps
            y = [0] * nsteps

            for pos, z in zip(positions, labels):
                mask[get_bin(pos)] = 1
                y[get_bin(pos)] = z

            X = discretizer.transform(X, end=T)[0]
            if (normalizer is not None):
                X = normalizer.transform(X)

            Xs.append(X)
            masks.append(np.array(mask))
            ys.append(np.array(y))

            assert np.sum(mask) > 0
            assert len(X) == len(mask) and len(X) == len(y)

        return [[Xs, masks], ys]

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.data[1])
                order = range(N)
                random.shuffle(order)
                tmp = [[[None]*N, [None]*N], [None]*N]
                for i in range(N):
                    tmp[0][0][i] = self.data[0][0][order[i]]
                    tmp[0][1][i] = self.data[0][1][order[i]]
                    tmp[1][i] = self.data[1][order[i]]
                self.data = tmp
            else:
                # sort entirely
                Xs = self.data[0][0]
                masks = self.data[0][1]
                ys = self.data[1]
                (Xs, masks, ys) = common_utils.sort_and_shuffle([Xs, masks, ys], B)
                self.data = [[Xs, masks], ys]

            for i in range(0, len(self.data[1]), B):
                X = self.data[0][0][i:i+B]
                mask = self.data[0][1][i:i+B]
                y = self.data[1][i:i+B]

                y_true = [np.array(x) for x in y]
                y_true = nn_utils.pad_zeros(y_true)
                y_true = np.expand_dims(y_true, axis=-1)

                if self.partition == 'log':
                    y = [np.array([metrics.get_bin_log(x, 10) for x in z]) for z in y]
                if self.partition == 'custom':
                    y = [np.array([metrics.get_bin_custom(x, 10) for x in z]) for z in y]

                X = nn_utils.pad_zeros(X) # (B, T, D)
                mask = nn_utils.pad_zeros(mask) # (B, T)
                y = nn_utils.pad_zeros(y)
                y = np.expand_dims(y, axis=-1)

                if self.return_y_true:
                    yield ([X, mask], y, y_true)
                else:
                    yield ([X, mask], y)

    def __iter__(self):
        return self.generator

    def next(self, return_y_true=False):
        with self.lock:
            self.return_y_true = return_y_true
            return self.generator.next()

    def __next__(self):
        return self.generator.__next__()
