import numpy as np
import random
from numba import njit

@njit
def swap_mutation(chrom: np.ndarray) -> np.ndarray:
    n = chrom.size
    if n < 2:
        return chrom

    i = random.randrange(n)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)

    chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


@njit
def polynomial_mutation(
    chrom: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    eta: float,
    p_mut: float = 1.0
) -> np.ndarray:

    d = chrom.size
    for j in range(d):
        if np.random.rand() > p_mut:
            continue

        y = chrom[j]
        yl, yu = low[j], high[j]
        delta1 = (y - yl) / (yu - yl)
        delta2 = (yu - y) / (yu - yl)

        rnd = np.random.rand()
        if rnd <= 0.5:
            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * ((1.0 - delta1) ** (eta + 1.0))
            deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
        else:
            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * ((1.0 - delta2) ** (eta + 1.0))
            deltaq = 1.0 - (val ** (1.0 / (eta + 1.0)))

        y_new = y + deltaq * (yu - yl)
        chrom[j] = max(min(y_new, yu), yl)

    return chrom