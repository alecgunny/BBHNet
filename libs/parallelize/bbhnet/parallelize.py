from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
from inspect import signature
from typing import Callable

from bbhnet.io.timeslides import Segment


class ProcessPool:
    def __init__(self, num_proc):
        self.executor = ProcessPoolExecutor(num_proc)

    def parallelize(self, f):
        return segment_iterator(f, self)

    def imap(self, f, segments, *args, **kwargs):
        futures = [
            self.executor.submit(f, i, *args, **kwargs) for i in segments
        ]
        try:
            for future in as_completed(futures):
                exc = future.exception()
                if exc is not None:
                    raise exc
                yield future.result()
        except Exception:
            self.executor.shutdown(wait=False, cancel_futures=True)
            raise


def segment_iterator(f: Callable, ex: ProcessPool) -> Callable:
    """
    Function wrapper that can parallelize a function across multiple segments
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        # get the name of the first argument, which should
        # be either a segment or a list of segments.
        # TODO: enforce naming or annotation constrictions here?
        arg0 = list(signature(f).parameters)[0]
        if len(args) == 0:
            # no args were passed so try to pop the argument
            # from the kwargs dictionary
            segments = kwargs.pop(arg0)
        else:
            # otherwise "pop" it from args
            segments = args[0]
            args = args[1:]

        if isinstance(segments, Segment):
            return f(segments, *args, **kwargs)
        else:
            # if we have multiple, analyze them in parallel
            return ex.imap(f, segments, *args, **kwargs)

    return wrapper
