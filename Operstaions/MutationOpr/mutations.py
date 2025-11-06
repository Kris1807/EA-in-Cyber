# Mutations per gene (1/N) with 50/50 inversion/swap
def mutate_per_gene(child):
    n = len(child)
    p = 1/n
    for i in range(n):
        if random.random() < p:
            if random.random() > 0.5:
                a = i
                b = random.randrange(n)
                if a > b: a, b = b, a
                if a < b:
                    child[a:b+1] = reversed(child[a:b+1])
            else:
                j = random.randrange(n)
                while j == i:
                    j = random.randrange(n)
                child[i], child[j] = child[j], child[i]

#example:
# mutate_per_gene(c1)
# mutate_per_gene(c2)
