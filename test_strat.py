from hittingtime import *
import numpy as np
from catan import Catan, CatanException, get_random_dice_arrangement, Player, simulate_game, simulate_game_and_save
from itertools import repeat
from catanAction import action, planBoard


##############################################


def not_safe(board):
    scored = []
    for x in range(5):
        for y in range(5):
            avg = average_resources_per_turn(board, [(x,y)])
            if avg[0]==0 or avg[1]==0 or avg[2]==0:
                pass
            else:
                return False
            scored.append((avg, (x,y)))
    return True

def make_board():
    width, height = 4, 4
    dice = get_random_dice_arrangement(width, height)
    resources = np.random.randint(0, 3, (height, width))
    return Catan(dice, resources)

def make_board_safe():
    def make_board():
        width, height = 4, 4
        dice = get_random_dice_arrangement(width, height)
        resources = np.random.randint(0, 3, (height, width))
        return Catan(dice, resources)
    board = make_board()
    while not_safe(board):
        board = make_board()
    return board

if __name__ == "__main__":
    from scipy import stats
    from time import time

    width, height = 4, 4
    dice = get_random_dice_arrangement(width, height)
    resources = np.random.randint(0, 3, (height, width))
    board = Catan(dice, resources)
    N = 1000

    def main(boards):
        t = time()
        trials = []
        i = 0
        for board in boards:
            trials.append(simulate_game(action, planBoard, board, 1))
            i += 1
            print("Turns taken in game %d: %d" % (i,trials[-1]))
        trials = np.array(trials)
        e = time()
        print("\nFinished in", time()-t, "seconds\n")
        print(stats.describe(trials))
    import cProfile
    main(make_board() for _ in range(N))
