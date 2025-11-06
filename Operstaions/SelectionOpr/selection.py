#binary selection of the better parent out of two random
def select_binary(pop, fits):
    i, j = random.randrange(len(pop)), random.randrange(len(pop))
    return pop[i][:] if fits[i] <= fits[j] else pop[j][:]

#example:
# p1 = select_binary(pop, fits)
# p2 = select_binary(pop, fits)