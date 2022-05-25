import numpy as np
import pytest

from bbhnet.analysis.normalizers import GaussianNormalizer


@pytest.fixture(
    params=[
        16,
        17,
        160,
        16.0,
        pytest.param(16.2, marks=pytest.mark.xfail(raises=ValueError)),
    ]
)
def norm_size(request):
    return request.param


def test_gaussian_normalizer(
    norm_size, window_size, boxcar_integration_test_fn
):
    normalizer = GaussianNormalizer(norm_size)

    # silly check but doing it solely
    # for the sake of checking int-ification
    assert normalizer.norm_size == int(norm_size)

    y = np.arange(1000)
    with pytest.raises(ValueError) as exc_info:
        normalizer(y, window_size)
    assert str(exc_info.value) == "GaussianNormalizer hasn't been fit"

    normalizer.fit(y)
    boxcar_integration_test_fn(normalizer.norm_size, normalizer.shifts)
    # TODO: check scale, check __call__, check __call__ shape exception
