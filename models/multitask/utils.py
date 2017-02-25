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
