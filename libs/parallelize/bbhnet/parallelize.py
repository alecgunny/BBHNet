from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
from inspect import signature

from bbhnet.io.timeslides import Segment


def segment_iterator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            num_proc = kwargs.pop("num_proc")
        except KeyError:
            num_proc = None

        arg0 = list(signature(f).parameters)[0]
        if len(args) == 0:
            segments = kwargs.pop(arg0)
        else:
            segments = args[0]
            args = args[1:]

        if isinstance(segments, Segment):
            if num_proc is not None:
                raise ValueError(
                    "'num_proc' can't be specified if {} "
                    "is a single Segment".format(arg0)
                )
            return f(segments, *args, **kwargs)

        if num_proc is None or num_proc == 1:
            for segment in segments:
                yield f(segment, *args, **kwargs)
        else:
            ex = ProcessPoolExecutor(num_proc)
            futures = [
                ex.submit(f, segment, *args, **kwargs) for segment in segments
            ]
            try:
                for future in as_completed(futures):
                    exc = future.exception()
                    if exc is not None:
                        raise exc

                    yield future.result()
            finally:
                ex.shutdown(wait=False, cancel_futures=True)

    return wrapper
