from mimic3models import nn_utils
from mimic3models import common_utils
import threading

def read_chunk(reader, chunk_size):
    data = []
    labels = []
    ts = []
    header = None
    for i in range(chunk_size):
        (X, t, y, header) = reader.read_next()
        data.append(X)
        ts.append(t)
        labels.append(y)
    return (data, ts, labels, header)


def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]
    return data


class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer,
                 batch_size, steps):

        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.steps = steps
        self.chunk_size = min(10000, steps * batch_size)
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _generator(self):
        B = self.batch_size
        while True:
            self.reader.random_shuffle()
            (data, ts, labels, header) = read_chunk(self.reader, self.chunk_size)
            data = preprocess_chunk(data, ts, self.discretizer, self.normalizer)
            data = (data, labels)
            common_utils.sort_and_shuffle(data, B)

            for i in range(0, self.chunk_size, B):
                yield (nn_utils.pad_zeros(data[0][i:i+B]),
                       data[1][i:i+B])

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return self.generator.next()

    def __next__(self):
        return self.generator.__next__()
