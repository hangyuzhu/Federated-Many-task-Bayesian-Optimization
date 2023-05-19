# coding=utf-8
import numpy as np
from pyDOE import lhs


def init_samples(xlb, xub, normalized, size, d, iid, partitions):
    if iid:
        # LHS sampling
        result = lhs(d, samples=size)
    else:
        # undergo
        l = [np.random.uniform(x[0], x[1], size=size)
             for x in partitions]
        result = np.array(list(map(list, zip(*l))))

    if normalized:
        return result
    else:
        return result * (xub - xlb) + xlb


def uniform_samples(xlb, xub, normalized, size, d):
    if normalized:
        return np.random.uniform(low=0, high=1, size=(size, d))
    else:
        return np.random.uniform(low=xlb, high=xub, size=(size, d))
