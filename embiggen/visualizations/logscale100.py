import functools
from matplotlib.colors import _make_norm_from_scale, Normalize
from matplotlib.scale import LogScale
import numpy as np


class LogScale100(LogScale):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, base=100, **kwargs)


@_make_norm_from_scale(functools.partial(LogScale100, nonpositive="mask"))
class LogNorm100(Normalize):
    """Normalize a given value to the 0-1 range on a log scale."""

    def autoscale(self, A):
        # docstring inherited.
        super().autoscale(np.ma.array(A, mask=(A <= 0)))

    def autoscale_None(self, A):
        # docstring inherited.
        super().autoscale_None(np.ma.array(A, mask=(A <= 0)))
