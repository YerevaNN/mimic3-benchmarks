import numpy as np
import os
import json

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
