from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mimic3models import common_utils
import threading
import random
import os


class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer, batch_size,
                 small_part, target_repl, shuffle, return_names=False):
        self.batch_size = batch_size
        self.target_repl = target_repl
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_data(reader, discretizer, normalizer, small_part)

        self.steps = (len(self.data[0]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_data(self, reader, discretizer, normalizer, small_part=False):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if (normalizer is not None):
            data = [normalizer.transform(X) for X in data]
        ys = np.array(ys, dtype=np.int32)
        self.data = (data, ys)
        self.ts = ts
        self.names = names

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[None] * N, [None] * N]
                tmp_names = [None] * N
                tmp_ts = [None] * N
                for i in range(N):
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                    tmp_ts[i] = self.ts[order[i]]
                self.data = tmp_data
                self.names = tmp_names
                self.ts = tmp_ts
            else:
                # sort entirely
                X = self.data[0]
                y = self.data[1]
                (X, y, self.names, self.ts) = common_utils.sort_and_shuffle([X, y, self.names, self.ts], B)
                self.data = [X, y]

            self.data[1] = np.array(self.data[1])  # this is important for Keras
            for i in range(0, len(self.data[0]), B):
                x = self.data[0][i:i+B]
                y = self.data[1][i:i+B]
                names = self.names[i:i + B]
                ts = self.ts[i:i + B]

                x = common_utils.pad_zeros(x)
                y = np.array(y)  # (B, 25)

                if self.target_repl:
                    y_rep = np.expand_dims(y, axis=1).repeat(x.shape[1], axis=1)  # (B, T, 25)
                    batch_data = (x, [y, y_rep])
                else:
                    batch_data = (x, y)

                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names, "ts": ts}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


def save_results(names, ts, predictions, labels, path):
    n_tasks = 25
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        header = ["stay", "period_length"]
        header += ["pred_{}".format(x) for x in range(1, n_tasks + 1)]
        header += ["label_{}".format(x) for x in range(1, n_tasks + 1)]
        header = ",".join(header)
        f.write(header + '\n')
        for name, t, pred, y in zip(names, ts, predictions, labels):
            line = [name]
            line += ["{:.6f}".format(t)]
            line += ["{:.6f}".format(a) for a in pred]
            line += [str(a) for a in y]
            line = ",".join(line)
            f.write(line + '\n')
