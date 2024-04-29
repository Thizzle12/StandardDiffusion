from functools import wraps
from time import time

from standarddiffusion.misc.utils import detach_to_numpy


def detach_2_numpy(func):
    """Detaches output tensor of a given function and converts it to numpy. \n
    If tensor is on the gpu, it is put on the cpu. \n
    This decorator should only decorate functions that returns a tensor.

    Args:
    ------
        func: Takes any function that returns a single tensor.

    Returns:
    ------
        ndarray: This wrapper returns an ndarray for any function it decorates.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return detach_to_numpy(func(*args, **kwargs))

    return wrapper
