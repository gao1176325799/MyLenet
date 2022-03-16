#contain all the libs used in this project
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import chain
from torch._C import dtype
import collections
from itertools import repeat
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse
_pair = _ntuple(2, "_pair")
#* pair
# def _ntuple(n, name="parse"):
#     def parse(x):
#         if isinstance(x, collections.abc.Iterable):
#             return tuple(x)
#         return tuple(repeat(x, n))

#     parse.__name__ = name
#     return parse


# _single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
# _triple = _ntuple(3, "_triple")
# _quadruple = _ntuple(4, "_quadruple")