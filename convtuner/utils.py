import contextlib
import logging
import time
from collections import Counter
from functools import partial, reduce
from operator import mul

log = logging.getLogger(__name__)


class Once(set):
    def __call__(self, *x):
        return x not in self and (self.add(x) or True)


def gflops(conv, image):
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    groups = conv.groups
    padding = conv.padding
    dilation = conv.dilation
    kernel_size = list(conv.kernel_size)
    stride = conv.stride
    batch_size, _, *in_sizes = image.shape
    out_sizes = [
        (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
        for i, v in enumerate(in_sizes)]
    gflops = product([2,
                      batch_size,
                      groups,
                      out_channels // groups,
                      in_channels // groups] +
                     out_sizes +
                     kernel_size)
    if conv.bias is None:
        gflops -= product([batch_size, out_channels] + out_sizes)
    return gflops / 1000000000.0


timers = Counter()


@contextlib.contextmanager
def timer(name):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    timers[name] += t1 - t0


product = partial(reduce, mul)


def retry(fn, attempts=3):
    for attempt in range(attempts):
        try:
            return fn()
        except:
            log.exception("error retry loop")
            if attempt == attempts - 1:
                raise
