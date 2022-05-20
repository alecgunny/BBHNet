from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from bbhnet.parallelize import as_completed


def func(x):
    return x**2


@pytest.fixture(params=[list, dict])
def container(request):
    values = range(10)
    if request.param is list:
        return list(values)
    else:
        letters = "abcdefghij"
        return {i: [j] for i, j in zip(letters, values)}


@pytest.fixture(params=[ThreadPoolExecutor, ProcessPoolExecutor])
def pool_type(request):
    return request.param


def test_as_completed(container, pool_type):
    futures = []
    with pool_type(2) as ex:
        if isinstance(container, dict):
            futures = {
                i: [ex.submit(func, k)]
                for i, j in container.items()
                for k in j
            }
        else:
            futures = [ex.submit(func, i) for i in container]

        for result in as_completed(futures):
            if isinstance(container, dict):
                letter, value = result
                letters = sorted(container.keys())
                assert value == letters.index(letter) ** 2
            else:
                assert 0 <= result**0.5 <= 9
