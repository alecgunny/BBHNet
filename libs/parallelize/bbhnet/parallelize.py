from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError,
)
from concurrent.futures import as_completed as _as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Union

FutureList = Iterable[Future]


def _handle_future(future: Future):
    exc = future.exception()
    if exc is not None:
        raise exc
    return future.result()


def as_completed(futures: Union[FutureList, Dict[Any, FutureList]]):
    if isinstance(futures, dict):
        futures = {
            k: _as_completed(v, timeout=1e-3) for k, v in futures.items()
        }
        while len(futures) > 0:
            keys = list(futures.keys())
            for key in keys:
                try:
                    future = next(futures[key])
                except TimeoutError:
                    continue
                except StopIteration:
                    futures.pop(key)
                    continue
                else:
                    yield key, _handle_future(future)
    else:
        for future in _as_completed(futures):
            yield _handle_future(future)


@dataclass
class AsyncExecutor:
    workers: int
    thread: bool = True

    def __post_init__(self):
        self._executor = None

    def __enter__(self):
        if self.thread:
            self._executor = ThreadPoolExecutor(self.workers)
        else:
            self._exector = ProcessPoolExecutor(self.workers)
        return self

    def __exit__(self, *exc_args):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None

    def submit(self, *args, **kwargs):
        if self._executor is None:
            raise ValueError("AsyncExecutor has no executor to submit jobs to")
        return self._executor.submit(*args, **kwargs)

    def imap(self, f: Callable, it: Iterable, **kwargs: Any):
        if self._executor is None:
            raise ValueError("AsyncExecutor has no executor to submit jobs to")

        futures = [self.submit(f, i, **kwargs) for i in it]
        return as_completed(futures)
