import time
from functools import wraps
import logging

timing_log = logging.getLogger(__name__)
timing_log.setLevel(logging.DEBUG)
timing_log.addHandler(logging.StreamHandler())


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.monotonic()
        result = f(*args, **kw)
        te = time.monotonic()
        timing_log.debug(f"func:{f.__name__} Time: {te - ts}")
        return result

    return wrap