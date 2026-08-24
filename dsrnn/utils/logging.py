import logging
from functools import wraps
from time import time

from lightning.pytorch.utilities import rank_zero_only


def timing(f):
    """Print the execution time of a function call."""
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def timed_model_call(model, *args, **kwargs):
    """Call a model and print the time taken."""
    return model(*args, **kwargs)


def time_for_model_call(model, *args, **kwargs) -> float:
    """Measure the elapsed time for a single model call."""
    ts = time()
    _ = model(*args, **kwargs)
    te = time()
    return te - ts


def average_time_model_call(model, n_calls: int = 20, *args, **kwargs) -> float:
    """Compute the average time over multiple model calls (one warm-up)."""
    time_for_model_call(model, *args, **kwargs)

    times = []
    for _ in range(0, n_calls + 1):
        times.append(time_for_model_call(model, *args, **kwargs))

    return sum(times) / len(times)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initialize a multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
