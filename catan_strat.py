from hittingtime import *
import numpy as np
from catan import Catan, CatanException, get_random_dice_arrangement, Player, simulate_game, simulate_game_and_save
from itertools import repeat

TURNS_UNTIL_RECOMPUTE = 3

###################################



class Goal:
    def __init__(self, tasks=[], x=-1, y=-1):
        self.tasks = list(tasks)
        self.x, self.y = x, y

def resource_hitting_time(board, x, y):
    cur_resources = np.array(board.get_resources())
    if x >= 0 and y >= 0:
        for dx in [-1, 0]: 
            for dy in [-1, 0]:
                xx = x + dx
                yy = y + dy
                if board.is_tile(xx, yy): 
                    die = board.dice[yy, xx] 
                    resource = board.resources[yy, xx]
                    cur_resources[die - 2][resource] += 1
    task = Task()
    task.trade_costs = updated_trade_costs_with_settlement(board, task.trade_costs, x, y)
    def htime(b):
        task.resources_needed = b
        return hitting_time((0,0,0), task.resources_needed, cur_resources, trading_rule_tuple_from_task(task))[0]
    
    return htime((2, 2, 2)) # + 0.5 * htime((1, 1, 0)) + 0.2 * htime((0, 2, 2))

#Best hyperparams
# h1, h2: (3, 8) OR (7, 1) 

h1 = 3
h2 = 10
h3 = 5

def weight_function(htime, p, c, v):
    speed = (1 / (0.1+htime))
    prosperity = (1 - (v / 10.0)) / (0.1 + p)
    return h1 * speed + h2 * speed * prosperity + h3 * prosperity


class CityGoal(Goal):
    def estimate_weight(self, player, htime):
        global h1, h2 # you don't need the global keyword to use non-local variables, only to edit
        v = player.points
        p = resource_hitting_time(player.board, self.x, self.y)
        #c = resource_hitting_time(player.board, -1, -1)
        return C * weight_function(htime, p, 0, v)

class SettlementGoal(Goal):
    def estimate_weight(self, player, htime):
        global h1, h2
        v = player.points
        p = resource_hitting_time(player.board, self.x, self.y)
        #c = resource_hitting_time(player.board, -1, -1)
        return S * weight_function(htime, p, 0, v)


class PortGoal(SettlementGoal):
    def estimate_weight(self, player, htime):
        global h1, h2
        v = player.points
        p = resource_hitting_time(player.board, self.x, self.y)
        #c = resource_hitting_time(player.board, -1, -1)
        return P * weight_function(htime, p, 0, v)


class CardGoal(Goal):
    def estimate_weight(self, player, htime):
        v = player.points
        if v == 9 and htime < 1:
             return float("inf")
        if v == 8:
             return h1 * (1 / (0.1+htime))
        return 0.1


def trading_rule_tuple_from_task(task):
    return make_trading_rule_tuple(tuple(task.resources_needed), tuple(task.trade_costs))

# Takes in these tuple parameters and returns a tuple of encoded trading rules
from functools import lru_cache
memoize = lru_cache(1024)
@memoize
def make_trading_rule_tuple(end, trade_costs):
    return tuple(encode(*(static_trading_rule(decode(i), end, trade_costs)[0])) for i in range(343))

# copy of the trading rule for Task but with self removed
def static_trading_rule(start, resources_needed, trade_costs):
    w, b, g = start
    current = [w, b, g]
    do_trade = [None, None, None]
    for i in range(3):
      if current[i] >= resources_needed[i] + trade_costs[i]:
        diff = np.array(resources_needed) - np.array(current)
        trade_idx = np.argmax(diff)
        if diff[trade_idx] > 0 and current[trade_idx] < 6:
          current[i] -= trade_costs[i] # The bug was here, it subtracted 4 instead of trade_costs[i]
          current[trade_idx] += 1
          do_trade[i] = (i, trade_idx)
    return tuple(current), do_trade


class Task:
    resources_needed = (0, 0, 0)
    trade_costs = (4, 4, 4) 
    def __init__(self, x=-1, y=-1):
        self.x, self.y = x, y
    
    def make_trading_rule(self, player):
        return lambda w, b, g: self.trading_rule(w, b, g)[0]
        
    def trading_rule(self, w, b, g):
        return static_trading_rule((w,b,g), self.resources_needed, self.trade_costs)
    
    def execute_trade(self, player):
        w, b, g = player.resources
        result, trades = self.trading_rule(w, b, g)
        for i in range(3):
            if trades[i] != None:
                #print(trades, player.resources, self.trade_costs)
                player.trade(trades[i][0], trades[i][1])
    
    def execute(self, player):
        pass
    
    def test(self):
        return self.resources_needed

class CardTask(Task):
    resources_needed = (1, 2, 2)
    def execute(self, player):
        self.execute_trade(player)
        if player.if_can_buy("card"):
            player.buy("card")
            return True
        return False

class CityTask(Task):
    resources_needed = (0, 3, 3)
    def execute(self, player):
        self.execute_trade(player)
        if player.if_can_buy("city"):
            # print("building city at (%d, %d)!" % (self.x, self.y))
            player.buy("city", self.x, self.y)
            player.available_locations.add((self.x, self.y))
            return True
        return False

def decide_port_number_else_None(board, x, y):
    if board.is_port(encode_loc(x, y)):
        return board.which_port(encode_loc(x, y))
    return None


def new_trading_costs_with_port(trade_costs, port_num):
    if port_num == 3:
         return (min(trade_costs[0], 3), min(trade_costs[1], 3), min(trade_costs[2], 3))
    else:
         # yes this is a little silly, but I like tuples much more, and it only has three elements so who cares
         trade_costs = list(trade_costs)
         trade_costs[port_num] = 2
         return tuple(trade_costs)


def updated_trade_costs_with_settlement(board, trade_costs, x, y):
    port_num = decide_port_number_else_None(board, x, y)
    if port_num is not None:
         return new_trading_costs_with_port(trade_costs, port_num)
    return trade_costs


class SettlementTask(Task):
    resources_needed = (2, 1, 1)
    def execute(self, player):
        self.execute_trade(player)
        if player.if_can_buy("settlement"):
            # print("building settlement at (%d %d)!" % (self.x, self.y))
            player.buy("settlement", self.x, self.y)
            player.available_locations.add((self.x, self.y))
            Task.trade_costs = updated_trade_costs_with_settlement(player.board, Task.trade_costs, self.x, self.y)
            return True
        return False

class RoadTask(Task):
    resources_needed = (1, 1, 0)
    def execute(self, player):
        self.execute_trade(player)
        if player.if_can_buy("road"):
            # print("building road from", self.x, "to", self.y)
            player.buy("road", self.x, self.y)
            player.available_locations.add(self.x)
            player.available_locations.add(self.y)
            return True
        return False


#############################################


# this is a function pretty much copied from catan.py
# self here must be the board
def if_can_build_noroads(self, building, x, y): 
        """returns true if spot (x,y) is available, false otherwise"""
        if x< 0 or y<0 or x > self.width+1 or y > self.height + 1:
            raise CatanException("({0},{1}) is an invalid vertex".format(x,y))
        #first let's check that the spot is empty:
        if self.get_vertex_number(x,y) in self.cities or self.get_vertex_number(x,y) in self.settlements:
            return False

        ## upgrading first settlment into a city
        if (building == "city"):
            return self.get_vertex_number(x, y) in self.settlements

        ## If no cities, or settlements, build for freebies, otherwise need road connecting.
        for x1 in range(x-1,x+2):
            for y1 in range(y-1,y+2):
                if x1+y1 < x+y-1 or x1+y1 > x+y+1 or y1-x1 < y-x-1 or y1-x1 > y-x+1: ## only interested in up, down, left, and right
                    pass
                elif x1 < 0 or x1 > self.width or y1 < 0 or y1 > self.height: ## only interested in valid tiles
                    pass
                elif self.get_vertex_number(x1, y1) in self.settlements or self.get_vertex_number(x1, y1) in self.cities:
                    return False
        return True



##############################################


def encode_loc(x, y):
    return y * 5 + x

def decode_loc(i):
    return i % 5, i // 5

def hitting_time_for_a_road(player, roadTask):
    w, b, g = (1, 1, 0)
    trade_rule = trading_rule_tuple_from_task(roadTask)
    resources_per_roll = player.board.get_resources()
    exp,beta,indexes = hitting_time((0,0,0), (w,b,g), resources_per_roll, trade_rule)
    return exp # time from no resources

def approxeq(a, b):
    return abs(a-b) < 0.0001

def hitting_time_until_task(player, task, MEMO):
    if type(task) in MEMO:
        return MEMO[type(task)]
    
    if isinstance(task, RoadTask):
        MEMO[type(task)] = hitting_time_for_a_road(player, task)
        return MEMO[type(task)]
    
    w, b, g = task.resources_needed
    w_curr, b_curr, g_curr = player.resources
    if w_curr >= w and b_curr >= b and g_curr >=g:
        return 0
    trade_rule = trading_rule_tuple_from_task(task)
    resources_per_roll = player.board.get_resources()
    exp, beta, indexes = hitting_time(tuple(player.resources), tuple(task.resources_needed), resources_per_roll, trade_rule)
    MEMO[type(task)] = exp
    return MEMO[type(task)]

def calculate_hitting_time(player, goal, MEMO):
    return sum(hitting_time_until_task(player, task, MEMO) for task in goal.tasks)

def choose_goal_to_pursue(player, goals):
    weights = []
    MEMO = {}
    for goal in goals:
        time_to_goal = calculate_hitting_time(player, goal, MEMO)
        weight = goal.estimate_weight(player, time_to_goal)
        weights.append(weight)
    return goals[np.argmax(weights)]

def distance(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

def get_shortest_path_to_location(available_locations, x, y):
    """ Returns a list of ((x1, y1), (x2, y2)) tuples, each symbolizing a road to be built"""
    # this is the closest point to the destination (x, y)
    start = min(available_locations, key=lambda loc: distance(loc, (x, y)))
    # right now we assume that the best path to take is by going diagonally across the board
    # that way we maximize new available locations that have 4 neighboring resources
    # maybe ideally it would take into account where the other roads are and build in an opposite direction to them or smth
    amount_to_go = [x - start[0], y - start[1]] # (amount_to_go_right, amount_to_go_up)
    points_to_hit = [start]
    curr = list(start) # so that it is mutable, will need to cast to tuple before recording
    end  = [x, y]
    while curr != end:
        # if i == 0, we are moving left/right, if i == 1, we are moving down/up
        # this will move horizontally, then diagonally
        i = 0 if abs(amount_to_go[0]) >= abs(amount_to_go[1]) else 1
        amount = amount_to_go[i] // abs(amount_to_go[i])
        curr[i] += amount
        amount_to_go[i] -= amount
        points_to_hit.append(tuple(curr))
    # Now points_to_hit is populated with (x,y) tuples of all the locations we're going to hit
    return list(zip(points_to_hit, points_to_hit[1:]))

def generate_road_tasks_to_point(player, x, y):
    """If all building/road buys were made through the Task API, then 
            available_locations is a set of (x,y) tuples with points we can build roads from"""
    cheapest_roads_to_get_there = get_shortest_path_to_location(player.available_locations, x, y)
    tasks = []
    for road in cheapest_roads_to_get_there:
        tasks.append(RoadTask(road[0], road[1]))
    return tasks

def generate_settlement_goal_at_location(player, x, y):
    tasks = generate_road_tasks_to_point(player, x, y)
    tasks.append(SettlementTask(x, y))
    return SettlementGoal(tasks, x, y)

def generate_port_goal_at_location(player, x, y):
    tasks = generate_road_tasks_to_point(player, x, y)
    tasks.append(SettlementTask(x, y))
    return PortGoal(tasks, x, y)

def generate_all_settlement_port_goals(player):
    stlments = set(player.board.settlements)
    cities = set(player.board.cities)
    # all possible places to place new settlements
    possible = set((x,y) 
                for x in range(5) for y in range(5)
                if if_can_build_noroads(player.board, "settlement", x, y))
    ports = set([(0,0), (0,4), (4,0), (4,4)])
    goals = []
    for x, y in possible:
        if (x,y) in ports:
            goals.append(generate_port_goal_at_location(player,x,y))
        else:
            goals.append(generate_settlement_goal_at_location(player,x,y))
    return goals

def generate_all_city_goals(player):
    goals = []
    all_settlements = player.board.settlements
    for x, y in map(decode_loc, all_settlements):
        task = CityTask(x, y)
        goals.append(CityGoal([task], x, y))
    return goals

def generate_all_possible_goals(player):
    # Add all the possible settlment only goals (no building of roads)
    settlements = generate_all_settlement_port_goals(player)
    # Add all the possible city goals
    cities = generate_all_city_goals(player)
    # Add all the card goal
    cards = [
        CardGoal([CardTask()])
    ]
    return settlements + cards + cities

def action(self):
    if min(self.resources) < 0 or max(self.resources) > 6:
        print("FUCK")
        raise Exception()
    if self.turn_counter == 0:
        # What are we to do on our first turn?
        Task.trade_costs = (4, 4, 4)
        x, y = self.preComp
        self.available_locations = set()
        settlementTask = SettlementTask(x, y)
        settlementTask.execute(self)
        # initialize some values
        self.turns_to_recompute = 0
    
    if self.turns_to_recompute <= 0:
        # Let us compute our current goal
        possible_goals = generate_all_possible_goals(self)
        self.current_goal = choose_goal_to_pursue(self, possible_goals)
        self.turns_to_recompute = TURNS_UNTIL_RECOMPUTE
    
    tasks = self.current_goal.tasks
    while len(tasks) > 0:
        curr_task = tasks[0]
        if curr_task.execute(self):
            tasks.pop(0)
        else:
            break # Could not complete our task for some reason
    if len(tasks) == 0:
        self.turns_to_recompute = 0
    self.turns_to_recompute -= 1

def average_resources_per_turn(board, locations):
    DICE_ROLL_PROBS = 1/36 * np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    r = [0,0,0]
    for x, y in locations:
        for dx in [-1, 0]: 
            for dy in [-1, 0]: 
                xx = x + dx
                yy = y + dy
                if board.is_tile(xx, yy):
                    die = board.dice[yy, xx] 
                    resource = board.resources[yy, xx]
                    r[resource] += 1 * DICE_ROLL_PROBS[die - 2]
    return r
                    
def planBoard(board):
    # Init
    Task.trade_costs = (4, 4, 4)
    task = Task()
    hitting_times = []
    better_hitting_times = []
    for x in range(5):
        for y in range(5):
            ht = resource_hitting_time(board, x, y)
            hitting_times.append((ht, (x,y)))
            w, b, g = average_resources_per_turn(board, [(x,y)])
            if w > 0 and b > 0 and g > 0:
                better_hitting_times.append((ht, (x,y)))
    return min(better_hitting_times)[1] if better_hitting_times else min(hitting_times)[1]


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

    def main(n):
        t = time()
        num_trials = n
        trials = []
        for i in range(n):
            board = make_board()
            trials.append(simulate_game(action, planBoard, board, 1))
            # print(i, trials[-1], np.mean(np.array(trials)), np.std(np.array(trials)))
        trials = np.array(trials)
        e = time()
        print("\nFinished in", time()-t, "seconds\n")
        #print(trials,"\n")
        print(stats.describe(trials))
        print()
        return stats.describe(trials)

from random import shuffle
K = [(a,b,c,d,e,f) for a in range(1,4) for b in range(4) for c in range(4) for d in range(1,4) for e in range(1,4) for f in range(1,4)]
shuffle(K)
for arg in K:
            h1, h2, h3, C, S, P = arg
            print(arg)
            out = main(500)
            with open("out.txt", "a") as f:
                 f.write("Using" +str(arg)+ "\n" + str(out) + "\n")
import cProfile
cProfile.run("main(250)")

