import numpy as np
import random
from numba import njit

def one_point_order_crossover(p1, p2):
    n = len(p1)
    cut = random.randrange(1, n)  # 1..n-1
    head1 = np.array(p1[:cut])
    tail1 = np.array([g for g in p2 if g not in head1])
    c1 = np.concatenate((head1, tail1))
    head2 = np.array(p2[:cut])
    tail2 = np.array([g for g in p1 if g not in head2])
    c2 = np.concatenate((head2, tail2))
    return c1.tolist(), c2.tolist()


@njit
def sbx_crossover(p1, p2, low, high, eta):
    d = p1.size
    c1 = np.empty_like(p1)
    c2 = np.empty_like(p2)
    for j in range(d):
        if p1[j] == p2[j]:
            c1[j] = p1[j]
            c2[j] = p2[j]
            continue
        if p1[j] < p2[j]:
            x1, x2 = p1[j], p2[j]
        else:
            x1, x2 = p2[j], p1[j]
        #This accounts for error that could occur with zero divisors
        dx = x2 - x1
        if abs(dx) < 1e-14:
            c1[j] = p1[j]
            c2[j] = p2[j]
            continue
        rand = np.random.rand()
        beta = 1.0 + (2.0 * (x1 - low[j]) / dx)
        alpha = 2.0 - beta ** (-(eta + 1.0))

        if rand <= 1.0/alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * dx)

        beta = 1.0 + (2.0 * (high[j] - x2) / dx)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0/alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * dx)

        if p1[j] < p2[j]:
            c1[j], c2[j] = child1, child2
        else:
            c1[j], c2[j] = child2, child1

        c1[j] = np.clip(c1[j], low[j], high[j])
        c2[j] = np.clip(c2[j], low[j], high[j])
    return c1, c2
