import numpy as np

def action(self):
    if self.board.settlements == []:
        (x,y) = self.preComp 
        self.buy("settlement", x, y)
    elif self.if_can_buy("card"):
        self.buy("card")
    elif self.resources[np.argmax(self.resources)] >= 4:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)
    return
'''
def planBoard(baseBoard):
    x = genRand(0,baseBoard.width+1)
    y = genRand(0,baseBoard.height+1)
    optSettlementLoc = (x,y)
    return optSettlementLoc
'''
def planBoard(baseBoard):
    print ("plan board")
    resources_per_tile = {}
    for v in range(baseBoard.max_vertex):
        loc = baseBoard.get_vertex_location(v)
        resources_per_tile[loc] = (set(), 0)
        for i in [-1, 0]:
            for j in [-1,0]:
                neighbor = (loc[0]+i, loc[1] + j)
                if neighbor.is_tile():
                    resources_per_tile[loc][0].add(baseBoard.resources[neighbor])
                    resources_per_tile[loc][1] += 1
    optimal_loc = max(resources_per_tile.keys(), key = lambda x: len(resources_per_tile[x][0]) + resources_per_tile[x][1])


def genRand(low,high):
    return np.random.randint(low, high)