import math


def calc_shifts_required(Tb: float, T: float, delta: float) -> int:
    r"""
    Calculate the number of shifts required to generate Tb
    seconds of background.

    The algebra to get this is gross but straightforward.
    Just solving
    $$\sum_{i=1}^{N}(T - i\delta) \geq T_b$$
    for the lowest value of N, where \delta is the
    shift increment.

    TODO: generalize to multiple ifos and negative
    shifts, since e.g. you can in theory get the same
    amount of Tb with fewer shifts if for each shift
    you do its positive and negative. This should just
    amount to adding a factor of 2 * number of ifo
    combinations in front of the sum above.
    """

    discriminant = (delta / 2 - T) ** 2 - 2 * delta * Tb
    N = (T - delta / 2 - discriminant**0.5) / delta
    return math.ceil(N)


def _intify(x: float):
    """
    Converts the input float into an int if the two are equal (e.g., 4.0 == 4).
    Otherwise, returns the input unchanged.
    """
    return int(x) if int(x) == x else x


def make_fname(prefix, t0, length):
    """Creates a filename for background files in a consistent format"""
    t0 = _intify(t0)
    length = _intify(length)
    return f"{prefix}-{t0}-{length}.hdf5"
