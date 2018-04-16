import numpy as np
import hello

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


def hitting_time_lst_calc(P, w_needed, b_needed, g_needed):
    ### your code here
    P -= np.eye(343)
    neg_one = -np.ones(343)
    for w in range(w_needed, 7):
        for b in range(b_needed, 7):
            for g in range(g_needed, 7):
                j = encode(w,b,g)
                P[:,j] = 0
                # P[j,:] = 0
                neg_one[j] = 0
    beta = np.linalg.lstsq(P, neg_one)[0]
    return None, beta



def first(to_at_least_wbg):
    include = []
    wg,bg,gg = to_at_least_wbg
    for w in range(0,7):
        for b in range(0,7):
            for g in range(0,7):
                if not (w>=wg and b>=bg and g>=gg):
                    include.append((w,b,g))
    return include


def build_trans_matrix(resources_per_roll, indexes, trade_rule):
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    # build the transition matrix
    sixes = np.array([6,6,6])
    print(resources_per_roll)
    resources_per_roll = resources_per_roll.astype(int)
    n = len(indexes)
    P = np.zeros((n,n))
    for i in indexes:
        wbg = np.array(decode(i))
        w,b,g = wbg
        for k, recv_resources in enumerate(resources_per_roll):
            wbg_new = np.minimum(wbg+recv_resources, sixes)
            wnew,bnew,gnew = trade_rule(*wbg_new)
            # wnew,bnew,gnew = min(6,wnew),  min(6,bnew),  min(6,gnew)
            j = encode(wnew,bnew,gnew)
            if j in indexes:
                P[indexes[i]][indexes[j]] += DICE_ROLL_PROBS[k]
    return P 


def build_trans_matrix(resources_per_roll, indexes, trade_rule):
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    # build the transition matrix
    sixes = np.array([6,6,6])
    resources_per_roll = resources_per_roll.astype(int)
    n = len(indexes)
    P = np.zeros((n,n))
    for wbg in indexes:
        w,b,g = wbg
        for k, recv_resources in enumerate(resources_per_roll):
            wr,br,gr = recv_resources
            wn,bn,gn = min(6,w+wr),min(6,b+br), min(6,g+gr)
            wn,bn,gn = trade_rule(wn,bn,gn)
            # wnew,bnew,gnew = min(6,wn),  min(6,bn),  min(6,gn)
            wbg_n = (wn,bn,gn)
            if wbg_n in indexes:
                P[indexes[wbg]][indexes[wbg_n]] += DICE_ROLL_PROBS[k]
    return P 



def hitting_time(from_wbg, to_at_least_wbg, resources_per_roll, trade_rule=lambda x,y,z:(x,y,z)):
    fw,fb,fg = from_wbg
    tw,tb,tg = to_at_least_wbg
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    include = first(to_at_least_wbg)    
    # record an index for each state
    indexes = {}
    for i,e in enumerate(include):
        indexes[e] = i
    n = len(indexes)
    # solve
    P = build_trans_matrix(resources_per_roll, indexes, trade_rule)
    beta = solve(P, n)
    i = indexes[from_wbg]
    return beta[i], beta, indexes

def hitting_time_lst(start, end, resources, trade_rule=lambda x,y,z:(x,y,z)):
    P = transition_matrix(resources, trade_rule)
    W,B,G = end
    indexes, beta = hitting_time_lst_calc(P, W, B, G)
    w,b,g = start
    i = int(encode(*start))
    return beta[i], beta, None


def hitting_time_old(start, end, resources, trade_rule=lambda x,y,z:(x,y,z)):
    P = transition_matrix(resources, trade_rule)
    W,B,G = end
    indexes, beta = hitting_time2(P, W, B, G)
    w,b,g = start
    i = int(encode(*start))
    i = indexes.index(i)
    return beta[i], beta, None


def hitting_time_transition_matrix(resources, trade_rule=lambda x,y,z:(x,y,z)):
    P = np.zeros((343,343))
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    for i in range(343):
        w,b,g = decode(i)
        for k in range(len(resources)):
            recieved_resources = resources[k]
            wresc, bresc, gresc = received_resources
            wnew, bnew, gnew = min(6, w + int(wresc)), min(6, b + int(bresc)), min(6, g + int(gresc))
            wnew, bnew, gnew = trade_rule(wnew, bnew, gnew)
            # wnew, bnew, gnew = min(6,wnew), min(6,bnew), min(6,gnew)
            j = encode(wnew, bnew, gnew)
            P[i][j] += DICE_ROLL_PROBS[k]
    return P


def hitting_time_using_P(start, end, P):
    to_include = sorted(first(end))    
    include = np.array(to_include)
    Z = P[np.ix_(include, include)]
    n = len(include)
    beta = solve(Z, n)
    j = int(encode(*start))
    return beta[to_include.index(j)], beta, to_include

def solve(P, n):
    P -= np.eye(n)
    neg_one = -np.ones(n)
    return np.linalg.solve(P, neg_one)


def should_include(to_at_least_wbg):
    include = []
    wg,bg,gg = to_at_least_wbg
    for w in range(0,7):
        for b in range(0,7):
            for g in range(0,7):
                if not (w>=wg and b>=bg and g>=gg):
                    include.append(encode(w,b,g))
    return include

def select_section(P, include):
    return P[np.ix_(include, include)]

def make_trans_matrix(resources, trade_rule, to_include):
    P = np.zeros(343*343)
    to_include = np.array(to_include)
    hello.hello_numpy(resources.reshape((33,)), np.array(range(343)), P)
    P = P.reshape((343,343))
    return P
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    P = np.zeros((343, 343))

    for i in to_include:
        w,b,g = decode(i)
        for k, received_resources in enumerate(resources):
            wresc, bresc, gresc = received_resources
            wnew, bnew, gnew = min(6, w + int(wresc)), min(6, b + int(bresc)), min(6, g + int(gresc))
            wnew, bnew, gnew = trade_rule(wnew, bnew, gnew)
            j = encode(wnew, bnew, gnew)
            P[i][j] += DICE_ROLL_PROBS[k]
    return P


def hitting_time_best(start, end, resources, trade_rule=lambda x,y,z: (x,y,z)):
    to_include = sorted(should_include(end))
    P = make_trans_matrix(resources, trade_rule, to_include)
    include = np.array(to_include)
    Z = select_section(P, include)
    n = len(include)
    beta = solve(Z, n)
    j = int(encode(*start))
    return beta[to_include.index(j)], beta, to_include




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
    W,B,G = (6,6,6)
    x, beta, indexes = hitting_time((0,0,0), (W,B,G), resources) # calculates time until we have six woods
    print(x, end=" ")
    x, beta, indexes = hitting_time_old((0,0,0), (W,B,G), resources)
    print(x, end=" ")
    x, beta, indexes = hitting_time_lst((0,0,0), (W,B,G), resources)
    print(x, end=" ")
    x, beta, indexes = hitting_time_best((0,0,0), (W,B,G), resources)
    print(x)


def mainN(n):
    for _ in range(n):
        main()

import cProfile
if __name__ == "__main__":
    resources = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
    print(list(make_trans_matrix(resources, lambda x,y,z:(x,y,z), list(range(343))).reshape(343*343)))
    cProfile.run("mainN(1000)")
