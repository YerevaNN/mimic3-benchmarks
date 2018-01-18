import numpy as np


def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)


def pad_zeros_from_left(arr):
    """
    `arr` is an array of `np.array`s

    The function appends zeros from left to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([np.zeros((max_len - x.shape[0],) + x.shape[1:]), x], axis=0)
           for x in arr]
    return np.array(ret)
