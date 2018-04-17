import numpy as np
import fast_transition_matrix as ftm
from functools import lru_cache

# hand = (w, b, g)
LIMIT = 7 # can have 0-6 of each resource

# Not too sure if this gives a memory speedup... arithmetic vs memory? It doesn't matter toooo much, but it's only 343 memory slots so whatever
encode_memo = lru_cache(360)
@encode_memo
def encode(w, b, g):
    return LIMIT * LIMIT * w + LIMIT * b + g

decode_memo = lru_cache(360)
@decode_memo
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

# This was the first iteration of our hitting time code, it's fairly grossly ineffecient, 
# but we believe it works, so we have been using it for testing purposes against our faster code
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

# This is the non-cached version of the hitting time c-extension code, we might wish to use it for debuging purposes in the case that we find our cached to be erroring
def hitting_time_extension(start, end, resources, trade_rule_tuple):
    resources_flattened = resources.reshape((33,))
    P, D = ftm.populate_transition_matrix(resources_flattened, np.array(list(end)), trade_rule_arr)
    n = np.count_nonzero(D+1) # all invalid entries are -1, all valid are >=0
    beta = solve(P.reshape((n,n)), n)
    j = int(encode(*start))
    return beta[D[j]], beta, D


# Just for testing
def approxeq(a, b):
    return abs(a-b) < 0.0001

# testing for accuracy, if our c-extension ever deviates from our simple but ineffecient python code, we throw an exception
def hitting_time_test(start, end, resources, trade_rule_tuple):
    trade_rule_function = translate_tuple_into_trade_rule(trade_rule_tuple)
    exp_old, beta_old, indexes_old = hitting_time_old(   start, end, resources, trade_rule_function)
    exp, beta, indexes             = hitting_time_cached(start, end, resources, trade_rule_tuple)
    for i, e in enumerate(indexes_old):
        if not approxeq(beta_old[i], beta[indexes[e]]):
            print(beta_old[i], beta[indexes[e]])
            raise Exception()
    return exp, beta, indexes


# separating this into another function mainly makes it easy to view its compute time in a profiler
# given the transition matrix where all end states are removed, it computes hitting time
def solve(P, n):
    P -= np.eye(n)
    neg_one = -np.ones(n)
    return np.linalg.solve(P, neg_one)

# The inputs to this functions are tuples so that they can be cached properly, if they are not, all hell will break loose
# The size of 1024 was decided arbitrarily, and making it larger didn't seem to increase performance, so it stays as it is
beta_cache = lru_cache(1024)
@beta_cache
def calc_beta_extension(end_tuple, resources_flattened_tuple, trade_rule_tuple):
    # Our C-Extension expects numpy arrays for each of its inputs
    P, D = ftm.populate_transition_matrix(np.array(resources_flattened_tuple), np.array(end_tuple), np.array(trade_rule_tuple))
    # all invalid entries are -1, all valid are >=0
    n = np.count_nonzero(D+1)
    beta = solve(P.reshape((n,n)), n)
    return beta, D
import math

# This function is essentially a wrapper for calc_beta_extension which ensures that all of its arguments are hashable so that our lru_cache can 
def hitting_time_cached(start, end, resources, trade_rule_tuple):
    resources_flattened = resources.reshape((33,)) # C-Extension expects this to be a flattened array, it's not expensive and it's easier
    beta, indexes = calc_beta_extension(tuple(end), tuple(resources_flattened), trade_rule_tuple)
    j = int(encode(*start))
    return beta[indexes[j]], beta, indexes # The indexes array maps each encoded value of a w,b,g to the index of that value in beta


# This is terribly slow, but hopefully only has to be done once
NO_TRADING_TUPLE = tuple(range(343))
def translate_trade_rule_into_tuple(trade_rule=None):
    if not trade_rule:
        return NO_TRADING_TUPLE # defaults to no trading if trade_rule is Falsey
    return tuple(encode(*trade_rule(*decode(i))) for i in range(343))

# This should never actually be called, it is just for testing purposes in hitting_time_test so that it may simply accept a tuple instead of a function
def translate_tuple_into_trade_rule(trade_tule_tuple=NO_TRADING_TUPLE):
    return lambda x,y,z: decode(trade_tule_tuple[encode(x,y,z)])


# inward wrapper for tuple usage, will eventually become outward once the trade_rule actually outputs an array and becomes cached
def hitting_time_tuple(start, end, resources, trade_rule_tuple):
    # requires trade_rule_tuple to be a tuple of length 343 that maps encoded values of w,b,g to the encoded value of what w,b,g would be traded to
    return hitting_time_cached(start, end, resources, trade_rule_tuple)

# outward wrapper
def hitting_time(start, end, resources, trade_rule=None):
    # don't even bother this function with tese queries please
    if start[0] >= end[0] and start[1] >= end[1] and start[2] >= end[2]:
        return 0
    trade_rule_tuple = translate_trade_rule_into_tuple(trade_rule)
    return hitting_time_tuple(start, end, resources, trade_rule_tuple)



