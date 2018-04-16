import numpy as np

# hand = (w, b, g)
LIMIT = 7 # can have 0-6 of each resource

def encode(w, b, g):
    return LIMIT * LIMIT * w + LIMIT * b + g

def decode(n):
    x = n % (LIMIT * LIMIT)
    return (n // (LIMIT * LIMIT), x // LIMIT, x % LIMIT)

def transition_matrix(resources, trade_rule = lambda x,y,z:(x,y,z)):
    """returns matrix T[i, j] = P(transition from state i to state j)"""
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


def hitting_time2(P, w_needed, b_needed, g_needed):
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


def hitting_time(from_wbg, to_at_least_wbg, resources_per_roll, trade_rule=lambda x,y,z:(x,y,z)):
    fw,fb,fg = from_wbg
    tw,tb,tg = to_at_least_wbg
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    
    # decide which states are relavant. Start by assuming all are, then remove irrelavant ones
    include = set(range(343))
    for w in range(to_at_least_wbg[0],7):
        for b in range(to_at_least_wbg[1],7):
            for g in range(to_at_least_wbg[2],7):
                include.remove(encode(w,b,g))
    
    # record an index for each state
    include = sorted(list(include))
    indexes = {}
    for i,e in enumerate(include):
        indexes[e] = i
    
    # build the transition matrix
    n = len(indexes)
    P = np.zeros((n,n))
    for i in indexes:
        w,b,g = decode(i)
        for k, recv_resources in enumerate(resources_per_roll):
            wrec,brec,grec = recv_resources
            wnew,bnew,gnew = min(6,w+wrec),min(6,b+brec),min(6,g+grec)
            wnew,bnew,gnew = trade_rule(wnew,bnew,gnew)
            wnew,bnew,gnew = min(6,wnew),  min(6,bnew),  min(6,gnew)
            j = encode(wnew,bnew,gnew)
            if j in indexes:
                P[indexes[i]][indexes[j]] += DICE_ROLL_PROBS[k]
    # solve
    P -= np.eye(n)
    neg_one = -np.ones(n)
    beta = np.linalg.solve(P, neg_one)
    i = indexes[encode(*from_wbg)]
    return beta[i], beta, indexes


if __name__ == '__main__':
    resources = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
    def trade_rule(w, b, g):
        # we just want six wood
        if b >= 4:
            w, b = w+1, b-4
        if g >= 4:
            w, g = w+1, g-4
        return min(6,w), min(6,b), min(6,g)
    W,B,G = (6,0,0)
    x, beta, indexes = hitting_time((0,0,0), (W,B,G), resources, trade_rule) # calculates time until we have six woods
    print(x)
    P = transition_matrix(resources, trade_rule)
    indexes, beta = hitting_time2(P, W, B, G)
    print(beta[0])
