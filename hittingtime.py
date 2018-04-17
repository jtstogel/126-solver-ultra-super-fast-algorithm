import numpy as np
import fast_transition_matrix as ftm

# hand = (w, b, g)
LIMIT = 7 # can have 0-6 of each resource

def encode(w, b, g):
    return LIMIT * LIMIT * w + LIMIT * b + g

def decode(n):
    x = n % (LIMIT * LIMIT)
    return (n // (LIMIT * LIMIT), x // LIMIT, x % LIMIT)

def transition_matrix(resources, trade_rule = lambda x,y,z:(x,y,z)):
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    P = np.zeros((343, 343))
    for i in range(343):
        w, b, g = decode(i)
        for k, received_resources in enumerate(resources):
            wresc, bresc, gresc = received_resources
            wnew, bnew, gnew = min(6, w + int(wresc)), min(6, b + int(bresc)), min(6, g + int(gresc))
            wnew, bnew, gnew = trade_rule(wnew, bnew, gnew)
            j = encode(wnew, bnew, gnew)
            P[i][j] += DICE_ROLL_PROBS[k]
    return P

def delete_from_P_and_solve(P, w_needed, b_needed, g_needed):
    ### your code here
    P = P - np.eye(343)
    to_delete = []
    for w in range(w_needed, 7):
        for b in range(b_needed, 7):
            for g in range(g_needed, 7):
                to_delete.append(encode(w, b, g))
    for j in reversed(sorted(to_delete)):
        P = np.delete(P, j, axis=1)
        P = np.delete(P, j, axis=0)
    neg_one = -np.ones(343-len(to_delete))
    beta = np.linalg.solve(P, neg_one)
    indices = sorted(list(set(range(343))-set(to_delete)))
    return indices, beta


def hitting_time_old(start, end, resources, trade_rule=lambda x,y,z:(x,y,z)):
    P = transition_matrix(resources, trade_rule)
    W,B,G = end
    indexes, beta = delete_from_P_and_solve(P, W, B, G)
    w,b,g = start
    i = int(encode(*start))
    i = indexes.index(i)
    return beta[i], beta, indexes

"""
The following functions all are for hitting_time, which calls an extension we wrote in C to compute the hittingtime
"""

def solve(P, n):
    P -= np.eye(n)
    neg_one = -np.ones(n)
    return np.linalg.solve(P, neg_one)

NO_TRADING = np.array(range(343))
def hitting_time(start, end, resources, trade_rule=None):
    if trade_rule is None:
        trade_rule = NO_TRADING
    else:
        trade_rule = np.array([encode(*trade_rule(*decode(i))) for i in range(343)])
    P, D = ftm.populate_transition_matrix(resources, np.array(list(end)), trade_rule)
    n = np.count_nonzero(D+1) # all invalid entries are -1, all valid are >=0
    beta = solve(P.reshape((n,n)), n)
    j = int(encode(*start))
    return beta[D[j]], beta, D


def approxeq(a, b):
    return abs(a-b) < 0.0001


def hitting_time_test(start, end, resources, trade_rule=None):
    exp_old, beta_old, indexes_old = hitting_time_old(start, end, resources, (lambda x,y,z:(x,y,z)) if trade_rule is None else trade_rule)
    exp, beta, indexes             = hitting_time(    start, end, resources, trade_rule)
    for i, e in enumerate(indexes_old):
        if not approxeq(beta_old[i], beta[indexes[e]]):
            print(beta_old[i], beta[indexes[e]])
            raise Exception()
    return exp, beta, indexes



def main():
    resources = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
    resources = np.array(resources)
    def trade_rule(w, b, g):
        # we just want six wood
        if b >= 4:
            w, b = w+1, b-4
        if g >= 4:
            w, g = w+1, g-4
        return min(6,w), min(6,b), min(6,g)
    W,B,G = (6,0,0)
    x, beta, indexes = hitting_time((0,0,0), (W,B,G), resources, trade_rule) # calculates time until we have six woods
    print(x, end=" ")
    x, beta, indexes = hitting_time_old((0,0,0), (W,B,G), resources, trade_rule)
    print(x, end=" ")
    x, beta, indexes = hitting_time_lst((0,0,0), (W,B,G), resources, trade_rule)
    print(x)


def mainN(n):
    for _ in range(n):
        main()

import cProfile
if __name__ == "__main__":
    cProfile.run("mainN(100)")
