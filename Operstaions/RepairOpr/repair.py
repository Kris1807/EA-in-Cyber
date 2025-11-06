#repair if improves
def try_two_opt_repair(child, D):
    n = len(child)
    a, b = sorted(random.sample(range(n), 2))
    a2 = (a + 1) % n
    b2 = (b + 1) % n
    A, B = child[a], child[a2]
    C, Dd = child[b], child[b2]
    delta = (D[A][C] + D[B][Dd]) - (D[A][B] + D[C][Dd])
    if delta < 0.0:
        if a2 <= b:
            child[a2:b+1] = reversed(child[a2:b+1])
        else:
            seg = child[a2:] + child[:b+1]
            seg.reverse()
            m = len(child) - a2
            child[a2:] = seg[:m]
            child[:b+1] = seg[m:]
        return True
    return False

# example usage:
# try_two_opt_repair(c1, D)
# try_two_opt_repair(c2, D)