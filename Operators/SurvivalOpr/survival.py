#replace by binary tournament
#tour_length was the fitness function used
def replace_via_binary_tournament(child, pop, fits, D):
    i, j = random.randrange(len(pop)), random.randrange(len(pop))
    weaker_idx = i if fits[i] >= fits[j] else j
    child_fit = tour_length(child, D)
    if child_fit < fits[weaker_idx]:
        pop[weaker_idx] = child
        fits[weaker_idx] = child_fit
        return True, child_fit
    return False, fits[weaker_idx]

#example:
# replaced1, _ = replace_via_binary_tournament(c1, pop, fits, D)
# replaced2, _ = replace_via_binary_tournament(c2, pop, fits, D)