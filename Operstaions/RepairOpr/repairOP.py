import numpy as np

def try_local_repair(
    child,
    fitness_fn,
    maximize=True,
    binary_idxs=None,
    bounds=None,
    rng=None,
):
    """
    A local 1/n-probability repair operator.

    Similar to try_two_opt_repair for TSP:
    - Select genes with probability 1/n
    - Propose a local change
    - Accept only if fitness improves
    - Return True if any change was accepted

    Parameters
    ----------
    child : np.ndarray
        Individual to repair. Modified in-place.
    fitness_fn : callable
        fitness_fn(x) -> float
    maximize : bool
        If True, improvement means fitness increases.
    binary_idxs : iterable[int] or None
        Indices that should be treated as binary features (0/1 flip).
    bounds : list[tuple(float,float)] or None
        Per-gene (min,max) for real-valued sampling.
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    bool
        True if the repair improved the child, otherwise False.
    """

    if rng is None:
        rng = np.random.default_rng()

    child = np.asarray(child)
    n = len(child)

    # 1) choose genes to repair: P = 1/n
    mask = rng.random(n) < (1.0 / n)
    idxs = np.nonzero(mask)[0]

    # if no index selected, force one random gene
    if len(idxs) == 0:
        idxs = np.array([rng.integers(0, n)])

    binary_idxs = set(binary_idxs or [])

    # compute current fitness
    current_f = fitness_fn(child)
    improved = False

    for i in idxs:
        old_val = child[i]

        # 2) propose new value for that gene
        if i in binary_idxs:
            # flip 0 <-> 1
            new_val = 1.0 - old_val

        else:
            # real-valued gene
            if bounds is not None:
                lo, hi = bounds[i]
                new_val = rng.uniform(lo, hi)

                # avoid identical value due to sampling limits
                tries = 0
                while np.isclose(new_val, old_val) and tries < 5:
                    new_val = rng.uniform(lo, hi)
                    tries += 1
            else:
                # default fallback: small gaussian step
                new_val = old_val + rng.normal(0, 1)

        # temporarily apply the change
        child[i] = new_val

        # evaluate new fitness
        new_f = fitness_fn(child)

        # 3) keep only if beneficial
        better = new_f > current_f if maximize else new_f < current_f

        if better:
            current_f = new_f
            improved = True
        else:
            # revert change
            child[i] = old_val

    return improved