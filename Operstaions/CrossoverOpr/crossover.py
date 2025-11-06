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