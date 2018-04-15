import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math

### GLOBALS

SETTLEMENT = 0
CARD = 1
CITY = 2
ROAD = 3
MAX_POINTS = 10
MAX_RESOURCES = 6
START_RESOURCES = 3
LIMIT = 7

costs = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])

class CatanException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Catan:
    def __init__(self, dice, resources, settlements = [], cities = [], roads = []):
        self.width = dice.shape[1]
        self.height = dice.shape[0]
        self.dice = dice
        self.resources = resources
        self.settlements = settlements
        self.cities = cities
        self.roads = roads
        self.max_vertex = (self.width+1)*(self.height+1) - 1

    def is_port(self, vertex):
        return vertex == 0 or vertex == self.width or vertex == self.max_vertex or vertex == self.max_vertex - self.width

    ## 0 - 2:1 wood
    ## 1 - 2:1 brick
    ## 2 - 2:1 grain
    ## 3 - 3:1 general
    def which_port(self, vertex):
        if vertex == 0:
            return 0
        elif vertex == self.width:
            return 1
        elif vertex == self.max_vertex - self.width:
            return 2
        elif vertex == self.max_vertex:
            return 3
        else:
            raise CatanException("{0} is not a port".format(vertex))

    def get_vertex_number(self, x, y):
        return (self.height + 1) * y + x
    
    def get_vertex_location(self, n):
        return (n % (self.height+1), n // (self.height+1))
    
    def is_tile(self, x, y):
        """returns whether x,y is a valid tile"""
        return x >= 0 and x < self.width and y >= 0 and y < self.width

    def build_road(self, c0, c1):
        v0 = self.get_vertex_number(c0[0], c0[1])
        v1 = self.get_vertex_number(c1[0], c1[1])
        if self.if_can_build_road(v0, v1):
            self.roads.append((v0, v1))
        else:
            raise CatanException("({0},{1}) is an invalid road".format(c0, c1))
            
    def if_can_build_road(self, start, end):
        ##order the road vertices
        temp = max(start, end)
        v1 = min(start, end)
        v2 = temp
        """returns true if road is valid, false otherwise"""
        #check if road vertices are on the map
        if v1 < 0 or v2 < 0 or v1 > self.max_vertex or v2 > self.max_vertex:
            raise CatanException("({0},{1}) is an invalid road".format(v1, v2))
        if v1 == v2: return False
        #first let's check that the spot is empty:
        if (v1, v2) in self.roads or (v2, v1) in self.roads:
            return False
        
        #now let's check if the proposed road is valid.
        #CORNER CASES
        if v1 == 0 or v2 == 0:
            if not (v1 + v2 == 1 or v1 + v2 == self.width+1):
                return False
        if v1 == self.width or v2 == self.width:
            if not (v1 + v2 == 2*self.width - 1 or v1 + v2 == 3*self.width+ 1):
                return False
        if v1 == (self.width + 1)*self.height or v2 == (self.width + 1)*self.height:
            if not (v1 + v2 == 2*(self.width + 1)*self.height + 1 or v1 + v2 == (self.width + 1)*(2*self.height - 1)):
                return False
        if v1 == self.max_vertex or v2 == self.max_vertex:
            if not (v1 + v2== 2*self.max_vertex - 1 or v1 + v2== (2 * self.max_vertex - (self.width + 1))):
                return False
        #EDGE CASES... literally --
        ## left edge
        if v1%(self.width + 1) == 0 or v2%(self.width + 1) == 0:
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        ## bottom edge
        if v1 in range(1, self.width + 1) or v2 in range(1, self.width + 1):
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        ## right edge
        if v1 in range(self.width, self.max_vertex + 1, self.width + 1) or v2 in range(self.width, self.max_vertex + 1, self.width + 1):
            if not (v2 - v1 == self.width + 1 or (v2 - v1 and v2%(self.width + 1) != 0) == 1):
                return False
        ## top edge
        if v1 in range(self.max_vertex - self.width + 1, self.max_vertex) or v2 in range(self.max_vertex - self.width + 1, self.max_vertex):
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        #GENERAL CASE
        if not (v2 - v1 == self.width + 1 or v2 - v1 == 1): return False
        
        #If there are no roads, it must be connected to a settlement or a city
        if len(self.roads) == 0:
            if v1 not in self.settlements and v2 not in self.settlements and v1 not in self.cities and v2 not in self.cities:
                return False

        #Otherwise, it must be connected to another road
        elif len(self.roads) != 0:
            if v1 not in set([element for tupl in self.roads for element in tupl]) and v2 not in set([element for tupl in self.roads for element in tupl]):
                return False
        return True
        
    
    def build(self, x, y, building):
        """build either a city or a settlement"""
        if self.if_can_build(building, x, y):
            vertex = self.get_vertex_number(x, y)
            if building == "settlement":
                self.settlements.append(vertex)
            elif building == "city":
                if vertex not in self.settlements:
                    raise CatanException("A settlement must be built first.")
                self.cities.append(vertex)
                self.settlements.remove(vertex)
            else:
                raise CatanException("{0} is an unknown building. Please use 'city' or 'settlement'.".format(building))
        else:
            raise CatanException("Cannot build {0} here. Please check if_can_build before building".format(building))
            
    
    def if_can_build(self, building, x, y):
        """returns true if spot (x,y) is available, false otherwise"""
        if x< 0 or y<0 or x > self.width+1 or y > self.height + 1:
            raise CatanException("({0},{1}) is an invalid vertex".format(x,y))
        #first let's check that the spot is empty:
        if self.get_vertex_number(x,y) in self.cities:
            return False

        ## upgrading first settlment into a city
        if (building == "city"):
            return self.get_vertex_number(x, y) in self.settlements
        ## If no cities, or settlements, build for freebies, otherwise need road connecting.
        if (len(self.settlements) + len(self.cities) != 0 and self.get_vertex_number(x, y) not in set([element for tupl in self.roads for element in tupl])):
            return False
        for x1 in range(x-1,x+2):
            for y1 in range(y-1,y+2):
                if x1+y1 < x+y-1 or x1+y1 > x+y+1 or y1-x1 < y-x-1 or y1-x1 > y-x+1: ## only interested in up, down, left, and right
                    pass
                elif x1 < 0 or x1 > self.width or y1 < 0 or y1 > self.height: ## only interested in valid tiles
                    pass
                elif self.get_vertex_number(x1, y1) in self.settlements or self.get_vertex_number(x1, y1) in self.cities:
                    return False
        return True

    def get_resources(self):
        """Returns array r where:
        r[i, :] = resources gained from throwing a (i+2)"""
        r = np.zeros((11, 3)) 
        for vertex in self.settlements:
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        resource = self.resources[yy, xx]
                        r[die - 2, resource] += 1
        for vertex in self.cities:
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        resource = self.resources[yy, xx]
                        r[die - 2, resource] += 2
        return r
    
    def draw(self):
        print("Drawing...")
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(-0.02,self.width+0.02)
        ax.set_ylim(-0.02,self.height+0.02)
        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for x in range(self.width):
            for y in range(self.height):
                color = ["brown", "red", "green"][self.resources[y, x]]
                ax.add_patch(patches.Rectangle((x, y),1,1, 
                                               facecolor=color, ec = "black"))
                if self.dice[y,x] != 0:
                    ax.text(x+0.5, y+0.5, str(self.dice[y, x]), fontsize=15)
        ## draw roads
        for road in self.roads:
            x0, y0 = self.get_vertex_location(road[0])
            x1, y1 = self.get_vertex_location(road[1])
            #vertical road
            if x0 == x1:
                # ax.add_patch(patches.Rectangle((x0 - 0.05, (y0 + y1)/2 + 0.05), 0.1, 0.9,
                #                                facecolor = "white"))
                ax.add_patch(patches.Rectangle((x0 - 0.05, min(y0,y1) + 0.05), 0.1, 0.9,
                                               facecolor = "white", ec = "black"))
            #horizontal road
            elif y0 == y1:
                # ax.add_patch(patches.Rectangle(((x0 + x1)/2 + 0.05, y0 - 0.05), 0.9, 0.1,
                #                                 facecolor = "white"))
                ax.add_patch(patches.Rectangle((min(x0,x1) + 0.05, y0 - 0.05), 0.9, 0.1,
                                                facecolor = "white", ec = "black"))
        for vertex in self.settlements:
            x, y = self.get_vertex_location(vertex)
            ax.add_patch(patches.Rectangle((x-0.1, y-0.1),0.2,0.2, 
                                           facecolor="purple", ec = "black"))
            ax.text(x-0.1, y-0.09, "1", fontsize=15, color="white")
        for vertex in self.cities:
            x, y = self.get_vertex_location(vertex)
            ax.add_patch(patches.Rectangle((x-0.1, y-0.1),0.2,0.2, 
                                           facecolor="blue", ec = "black")) 
            ax.text(x-0.1, y-0.09, "2", fontsize=15, color="white")
            
            

class Player:    
    def __init__(self, action, preComputed, board, resources, points = 0, turn_counter = 0):
        self.board = board
        self.action = action
        self.preComp = preComputed
        self.resources = resources
        self.points = points
        self.turn_counter = turn_counter
        
    def if_can_buy(self, item):
        if item == "card":
            return np.all(self.resources >= costs[CARD,:])
        elif item == "settlement":
            return np.all(self.resources >= costs[SETTLEMENT,:])
        elif item == "city":
            return np.all(self.resources >= costs[CITY,:])
        elif item == "road":
            return np.all(self.resources >= costs[ROAD,:])
        else:
            raise CatanException("Unknown item: {0}".format(item))

    def buy(self, item, x=-1,y=-1):
        if item == "card":
            self.points += 1
            self.resources = np.subtract(self.resources,costs[1])
        elif item == "road": #input should be of format board.buy("road", (1,1), (1,2))
            v0 = self.board.get_vertex_number(x[0], x[1])
            v1 = self.board.get_vertex_number(y[0], y[1])
            if self.board.if_can_build_road(v0, v1):
                self.board.build_road(x, y)
                self.resources = np.subtract(self.resources, costs[ROAD,:])
        elif (item == "settlement" or item == "city") and self.board.if_can_build(item,x,y):
            self.board.build(x,y,item)
            if item == "settlement":
                self.points += 1
                self.resources = np.subtract(self.resources,costs[SETTLEMENT,:])
            else:
                self.points += 1
                self.resources = np.subtract(self.resources,costs[CITY,:])

    #Trading
    def trade(self, r_in, r_out):
        required = 4
        ports = []
        for e in self.board.settlements:
            if self.board.is_port(e):
                ports.append(self.board.which_port(e))
        for e in self.board.cities:
            if self.board.is_port(e):
                ports.append(self.board.which_port(e))
        if r_in in ports:
            required = 2
        if 3 in ports:
            required = min(required, 3)
        if self.resources[r_in] < required or self.resources[r_out] == MAX_RESOURCES:
            raise CatanException("Invalid trade.")
        self.resources[r_in] -= required
        self.resources[r_out] += 1
    
    def play_round(self):
        dice_roll = np.random.randint(1,7)+np.random.randint(1,7)

        # collect resources
        collected_resources = self.board.get_resources()[dice_roll-2,:]
        self.resources = np.add(self.resources,collected_resources)
        self.resources = np.minimum(self.resources, MAX_RESOURCES) # LIMIT IS MAX # OF RESOURCES
            
        # perform action
        self.action(self)
        assert np.max(self.resources) < LIMIT
        
        # update the turn counter
        self.turn_counter += 1
        
        return dice_roll

def simulate_game(action, planBoard, board, num_trials):
    """Simulates 'num_trials' games with policy 'action' and returns average length of games"""
    results = list()
    preComputed = planBoard(board)
    for _ in range(num_trials):
        resources = np.array([START_RESOURCES, START_RESOURCES, START_RESOURCES])
        live_board = Catan(board.dice, board.resources, [], [], [])
        player = Player(action, preComputed, live_board, resources)

        while player.points < MAX_POINTS:
            if player.turn_counter > 1000000:
                raise CatanException("possible infinite loop (over 1M turns)")
                break
            player.play_round()
        results.append(player.turn_counter)
    
    return np.sum(results)/float(num_trials)

def simulate_game_and_save(action, planBoard, board):
    """Simulates 1 game with policy 'action' and returns data about game"""
    results = list()
    preComputed = planBoard(board)
    resources = np.array([START_RESOURCES, START_RESOURCES, START_RESOURCES])
    live_board = Catan(board.dice, board.resources, [], [], [])
    player = Player(action, preComputed, live_board, resources)
    
    settlements = []
    cities = []
    roads = []
    hands = []
    live_points = []
    dice_rolls = []

    while player.points < MAX_POINTS:
        if player.turn_counter > 1000000:
            raise CatanException("possible infinite loop (over 1M turns)")
            break
        dice_roll = player.play_round()
        dice_rolls.append(dice_roll)
        settlements.append(live_board.settlements[:])
        cities.append(live_board.cities[:])
        roads.append(live_board.roads[:])
        hands.append(player.resources[:])
        live_points.append(player.points)
    
    return settlements, cities, roads, hands, live_points, dice_rolls

def get_random_dice_arrangement(width, height):
    """returns a random field of dice"""
    # return np.random.randint(2,13,(height,width))
    ns = list(range(2, 13)) * int(width * height / 10 + 1)
    np.random.shuffle(ns)
    ns = ns[:width*height]
    ns = np.reshape(ns, (height, width))
    return ns
