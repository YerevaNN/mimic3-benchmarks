import numpy as np

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


def load_mortalities(reader, discretizer, normalizer, small_part=False):
    N = reader.get_number_of_examples()
    if (small_part == True):
        N = 1000
    (data, ts, mortalities, header) = read_chunk(reader, N)
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]
    return (np.array(data), mortalities)
