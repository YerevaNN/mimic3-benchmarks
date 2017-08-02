import numpy as np
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


def process_data(data, partition):

    X = nn_utils.pad_zeros(data_raw[0])
    fms = data_raw[1][0]
    loss = data_raw[1][1]
    phs = data_raw[1][2]
    sws = data_raw[1][3]
        
    ihm_pos = np.array([x[0] for x in fms], dtype=np.int32)
    ihm_mask = np.array([x[1] for x in fms], dtype=np.int32)
    ihm_label = np.array([x[2] for x in fms], dtype=np.int32)

    los_mask = [np.array(x[0], dtype=np.int32) for x in loss]
    los_mask = nn_utils.pad_zeros(los_mask).astype(np.int32)

    # TODO: use partition argument here
    los_label = [np.array(x[1], dtype=np.float32) for x in loss]
    #los_label = nn_utils.pad_zeros(los_label).astype(np.int32) # for regression
    los_label = [np.array([self.get_bin(y, self.nbins) for y in x], dtype=np.int32)
                    for x in los_label]
    los_label = nn_utils.pad_zeros(los_label).astype(np.int32)

    ph_label = [np.array(x, dtype=np.int32) for x in phs]
    ph_label = nn_utils.pad_zeros(ph_label).astype(np.int32)

    decomp_mask = [np.array(x[0], dtype=np.int32) for x in sws]
    decomp_mask = nn_utils.pad_zeros(decomp_mask).astype(np.int32)

    decomp_label = [np.array(x[1], dtype=np.int32) for x in sws]
    decomp_label = nn_utils.pad_zeros(decomp_label).astype(np.int32)

    return (X, (ihm_pos, ihm_mask, ihm_label,
                los_mask, los_label,
                ph_label,
                decomp_mask, decomp_label))


class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer, partition,
                 batch_size, small_part):
        data_raw = load_data(reader, discretizer, normalizer, small_part)
        self.data = [None] * 2
        self.data[0] = data_raw[0]
        self.data[1] = data_raw[1:]
        self.partition = partition
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _generator(self):
        B = self.batch_size
        while True:
            self.data = common_utils.sort_and_shuffle(self.data, B)
            for i in range(0, len(self.data[0]), B):
                x = self.data[0][i:i+B]
                y = self.data[1][i:i+B]
                yield process_data((x, y), self.partition)

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return self.generator.next()

    def __next__(self):
        return self.generator.__next__()
