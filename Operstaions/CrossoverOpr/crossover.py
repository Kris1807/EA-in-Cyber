#crossover that yields two permutation children
def one_point_order_crossover(p1, p2):
    n = len(p1)
    cut = random.randrange(1, n)  # 1..n-1
    head1 = p1[:cut]
    tail1 = [g for g in p2 if g not in head1]
    c1 = head1 + tail1
    head2 = p2[:cut]
    tail2 = [g for g in p1 if g not in head2]
    c2 = head2 + tail2
    return c1, c2

#example usage:
# c1, c2 = one_point_order_crossover(p1, p2)


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
    eta: float
) -> Tuple[np.ndarray, np.ndarray]:
    d = p1.size
    c1 = np.empty_like(p1)
    c2 = np.empty_like(p2)
    for j in range(d):
        if p1[j] == p2[j]:
            c1[j] = p1[j]; c2[j] = p2[j]; continue
        x1, x2 = (p1[j], p2[j]) if p1[j] < p2[j] else (p2[j], p1[j])
        rand = rng.random()

        beta = 1.0 + (2.0 * (x1 - low[j]) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0/alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0/(eta + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (high[j] - x2) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0/alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0/(eta + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        if p1[j] < p2[j]:
            c1[j], c2[j] = child1, child2
        else:
            c1[j], c2[j] = child2, child1

        c1[j] = min(max(c1[j], low[j]), high[j])
        c2[j] = min(max(c2[j], low[j]), high[j])
    return c1, c2

#example:
# c1, c2 = sbx_crossover(p1, p2, low, high, rng, eta) 