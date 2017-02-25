import numpy as np

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


def load_phenotypes(reader, discretizer, normalizer, small_part=False):
    N = reader.get_number_of_examples()
    if (small_part == True):
        N = 1000
    (data, ts, ys, header) = read_chunk(reader, N)
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]
    ys = np.array(ys, dtype=np.int32)
    return (data, ys)
