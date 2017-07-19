from mimic3models import nn_utils
import random

def read_chunk(reader, chunk_size):
    data = []
    mortalities = []
    ts = []
    header = None
    for i in range(chunk_size):
        (X, t, y, header) = reader.read_next()
        data.append(X)
        ts.append(t)
        mortalities.append(y)
    return (data, ts, mortalities, header)


def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]
    return data


def shuffle_and_sort(data, batch_size):
    assert(len(data) == 2)
    data = zip(*data)
    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    size = len(head)
    mas = [head[i : i+batch_size] for i in range(0, size, batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail
    # NOTE: we assume that we will not use cycling in batch generator
    # so all examples in one batch will have more or less the same context lenghts

    data = zip(*data)
    return data


class BatchGen(object):

    def __init__(self, reader, discretizer=None, normalizer=None,
                 batch_size=64, steps=10):

        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.steps = steps
        self.generator = self._generator()

    def _generator(self):
        B = self.batch_size
        iterations = 0
        while True:
            if iterations % 100 == 0:
                self.reader.random_shuffle()

            chunk_size = 100 * B
            (data, ts, mortalities, header) = read_chunk(self.reader, chunk_size)
            data = preprocess_chunk(data, ts, self.discretizer, self.normalizer)
            data = (data, mortalities)
            shuffle_and_sort(data, B)

            for i in range(0, chunk_size, B):
                yield (nn_utils.pad_zeros(data[0][i:i+B]),
                       data[1][i:i+B])

            iterations += 1

    def __iter__(self):
        return self.generator

    def next(self):
        return self.generator.next()

    def __next__(self):
        return self.generator.__next__()
