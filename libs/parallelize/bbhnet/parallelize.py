from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
from inspect import signature
from typing import Callable

from bbhnet.io.timeslides import Segment


def serial_gen(f, segments, *args, **kwargs):
    for segment in segments:
        yield f(segment, *args, **kwargs)


def async_gen(executor, futures):
    try:
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                raise exc

            yield future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def segment_iterator(f: Callable) -> Callable:
    """
    Function wrapper that can parallelize a function across multiple segments
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        # first check if a number of processes was specified,
        # and pop it out of the kwargs since num_proc shouldn't
        # be an argument of the function itself
        try:
            num_proc = kwargs.pop("num_proc")
        except KeyError:
            num_proc = None

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
            # if this is just a single segment, then just call
            # the function as-is on the single segment
            if num_proc is not None:
                raise ValueError(
                    "'num_proc' can't be specified if {} "
                    "is a single Segment".format(arg0)
                )
            return f(segments, *args, **kwargs)
        elif num_proc is None or num_proc == 1:
            # otherwise if we have multiple segments but didn't
            # specify multiprocessing, just iterate through the
            # function's output on the segments
            # TODO: some sort of type checking to ensure that
            # this is an iterable of segments?
            return serial_gen(f, segments, *args, **kwargs)
        else:
            # last case is that we have multiple segments _and_
            # we specified multiprocessing. So start up a process
            # pool and submit the function and various segments to
            # the pool for async processing
            ex = ProcessPoolExecutor(num_proc)
            futures = [
                ex.submit(f, segment, *args, **kwargs) for segment in segments
            ]
            return async_gen(ex, futures)

    return wrapper
