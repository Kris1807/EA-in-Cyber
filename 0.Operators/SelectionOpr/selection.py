#binary selection of the better parent out of two random
def select_binary(pop, fits):
    i, j = random.randrange(len(pop)), random.randrange(len(pop))
    return pop[i][:] if fits[i] <= fits[j] else pop[j][:]

#example:
# p1 = select_binary(pop, fits)
# p2 = select_binary(pop, fits)

# GA building blocks
def tournament_select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[0]
    best_fit = fitness[best]
    for i in idxs[1:]:
        if fitness[i] < best_fit:
            best, best_fit = i, fitness[i]
    return best

#example
# p1_idx = tournament_select(fitness, tournament_k, rng)
# p2_idx = tournament_select(fitness, tournament_k, rng)