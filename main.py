## CS440 Spring 2021
## Final exam: Question 1
## Mustafa Sadiq (ms3035)

import numpy as np
np.set_printoptions(precision=5, suppress=True)

def create_best_policy_map(cost_map, landscape):
    best_policy = np.full((17, 5), 'X', dtype=str)

    best_policy[16][0] = 'G'
    
    for x in range(17):
        for y in range(5):
            lowest = -10000000000
            # move up
            if x-1 >= 0 and not (landscape[x][y] == 'X' or landscape[x][y] == 'G'): 
                if cost_map[x-1][y] > lowest:
                    best_policy[x][y] = '^'
                    lowest = cost_map[x-1][y]
            # move down
            if x+1 <= 17-1 and not (landscape[x][y] == 'X' or landscape[x][y] == 'G'): 
                if cost_map[x+1][y] > lowest:
                    best_policy[x][y] = 'V'
                    lowest = cost_map[x+1][y]
            # move left
            if y-1 >= 0  and not (landscape[x][y] == 'X' or landscape[x][y] == 'G'): 
                if cost_map[x][y-1] > lowest:
                    best_policy[x][y] = '<'
                    lowest = cost_map[x][y-1]
            # move right
            if y+1 <= 5-1 and not (landscape[x][y] == 'X' or landscape[x][y] == 'G'): 
                if cost_map[x][y+1] > lowest:
                    best_policy[x][y] = '>'
                    lowest = cost_map[x][y+1]

    return best_policy

def create_landscape():
    landscape = np.full([17, 5], ' ', dtype=str)
    landscape[0][0] = 'S'
    landscape[16][0] = 'G'

    for x in range(2, 7):
        landscape[x][0] = 'X'
    for x in range(10, 15):
        landscape[x][0] = 'X'
    for x in range(6, 11):
        landscape[x][4] = 'X'
    
    return landscape

def create_cost_map(ravine_reward):
    cost_map = np.zeros((17, 5), dtype=float)
    cost_map[16][0] = 0

    for x in range(2, 7):
        cost_map[x][0] = ravine_reward
    for x in range(10, 15):
        cost_map[x][0] = ravine_reward
    for x in range(6, 11):
        cost_map[x][4] = ravine_reward
        
    return cost_map


def get_values(landscape, location, cost_map, discount_rate):
    x = location[0]
    y = location[1]

    values = []

    # move up
    if x-1 >= 0: 
        if y-1 >= 0 and y+1 <= 5-1:
            values.append(-1 + discount_rate * ((0.8*cost_map[x-1][y]) + (0.1*cost_map[x][y-1]) + (0.1*cost_map[x][y+1])))
        elif y-1 >= 0:
            values.append(-1 + discount_rate * ((0.9*cost_map[x-1][y]) + (0.1*cost_map[x][y-1])))
        elif y+1 <= 5-1:
            values.append(-1 + discount_rate * ((0.9*cost_map[x-1][y]) + (0.1*cost_map[x][y+1])))

    # move down
    if x+1 <= 17-1: 
        if y-1 >= 0 and y+1 <= 5-1:
            values.append(-1 + discount_rate * ((0.8*cost_map[x+1][y]) + (0.1*cost_map[x][y-1]) + (0.1*cost_map[x][y+1])))
        elif y-1 >= 0:
            values.append(-1 + discount_rate * ((0.9*cost_map[x+1][y]) + (0.1*cost_map[x][y-1])))
        elif y+1 <= 5-1:
            values.append(-1 + discount_rate * ((0.9*cost_map[x+1][y]) + (0.1*cost_map[x][y+1])))

    # move left
    if y-1 >= 0: 
        if x-1 >= 0 and x+1 <= 17-1:
            values.append(-1 + discount_rate * ((0.8*cost_map[x][y-1]) + (0.1*cost_map[x-1][y]) + (0.1*cost_map[x+1][y])))
        elif x-1 >= 0:
            values.append(-1 + discount_rate * ((0.9*cost_map[x][y-1]) + (0.1*cost_map[x-1][y])))
        elif x+1 <= 17-1:
            values.append(-1 + discount_rate * ((0.9*cost_map[x][y-1]) + (0.1*cost_map[x+1][y])))

    # move right
    if y+1 <= 5-1: 
        if x-1 >= 0 and x+1 <= 17-1:
            values.append(-1 + discount_rate * ((0.8*cost_map[x][y+1]) + (0.1*cost_map[x-1][y]) + (0.1*cost_map[x+1][y])))
        elif x-1 >= 0:
            values.append(-1 + discount_rate * ((0.9*cost_map[x][y+1]) + (0.1*cost_map[x-1][y])))
        elif x+1 <= 17-1:
            values.append(-1 + discount_rate * ((0.9*cost_map[x][y+1]) + (0.1*cost_map[x+1][y])))

    return values

    

########################################################
def value_iteration(ravine_reward=-1000, print_results=False):
    landscape = create_landscape()
    cost_map = create_cost_map(ravine_reward)    

    discount_rate = 1
    delta = 100
    theta = 1e-100

    iteration = 0
    while True:
        iteration += 1
        delta = 0
        for x in range(17):
            for y in range(5):
                if landscape[x][y] == 'X' or landscape[x][y] == 'G':
                    continue
                v = cost_map[x][y]                          
                cost_map[x][y]  = max(get_values(landscape, (x, y), cost_map, discount_rate))
                delta = max(delta, abs(v - cost_map[x][y]))

        if delta < theta:
            if print_results == True:
                print('\n\nConverged at iteration:', iteration)
            break

    best_policy = create_best_policy_map(cost_map, landscape)

    if print_results == True:
        print('\n\nLandscape:\n\n', landscape)
        print('\n\nMinimal expected cost:\n\n', cost_map)        
        print('\n\nBest policy:\n\n', best_policy)

    return best_policy



####################### Bonus #################################

def bonus():
    best_policy = value_iteration(0)
    for ravine_reward in range(-1, -1000, -1):
        print(ravine_reward)
        new_best_policy = value_iteration(ravine_reward)
        if np.array_equal(best_policy, new_best_policy):
            value_iteration(ravine_reward-1, print_results=True)
            break
        else:
            best_policy = np.copy(new_best_policy)


#######################################################
value_iteration(ravine_reward=-1000, print_results=True)
# bonus()