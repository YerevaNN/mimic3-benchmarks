import numpy as np
import os
import json
import random

from mimic3models.feature_extractor import extract_features


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if (len(channel_info[channel]['possible_values']) != 0):
            ret[i-1] = map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1])
        ret[i-1] = map(lambda x: (float(x[0]), float(x[1])), ret[i-1])
    return ret


def extract_features_from_rawdata(chunk, header, period, features):
    with open(os.path.join(os.path.dirname(__file__), "channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def sort_and_shuffle(data, batch_size):
    """ Sort data by length, then make batches and shuffle them
        data is tuple (X, y)
    """
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
