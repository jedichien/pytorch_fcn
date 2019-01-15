import numpy as np

def iu(predict, target):
    values = []
    batch_size, n_class = target.shape[0], target.shape[1]
    for idx in range(batch_size):
        _iu = []
        for clsid in range(n_class):
            _overlap = target[idx, clsid, ...] + (predict == clsid)[idx]
            intersection = _overlap == 2
            union = np.clip(_overlap, 0, 1)
            _iu.append(intersection.sum()/(union.sum()+1e-30))
        values.append(_iu)
    values = np.asarray(values)
    means = np.mean(values, axis=1)
    return values, means