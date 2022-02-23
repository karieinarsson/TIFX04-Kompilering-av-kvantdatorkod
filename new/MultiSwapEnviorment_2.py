from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math
from typing import List, Tuple

# types
Matrix = List[List[int]]
Action = Tuple[int]

#Our enviorment
class SwapEnviorment(Env):
    def __init__(self, depth_of_code: int, rows: int, cols: int, 
                 max_swaps_per_time_step: int=-1, used: List[int]=None) -> None:
        self.depth_of_code = depth_of_code
        self.rows = rows
        self.cols = cols
        if used is None:
            used = []
        self.used = used
        if max_swaps_per_time_step < 0 or max_swaps_per_time_step > np.floor(self.rows*self.cols/2):
            self.max_swaps_per_time_step = np.floor(self.rows * self.cols/4)
        else: 
            self.max_swaps_per_time_step = max_swaps_per_time_step
        self.max_steps_per_episode = 200
        #array of possible actions
        self.possible_actions = self.get_possible_actions()
        #Number of actions we can take
        self.action_space = Discrete(len(self.possible_actions))
        print(self.action_space)
        self.observation_space = Box(low=0, high=math.floor(rows * cols/2),
                                shape=(depth_of_code*rows * cols, ), dtype=np.int)
        #The start state
        self.state = self.make_state()
        #max amount of layers per episode
        self.max_layers = self.depth_of_code


    def step(self, action: Discrete) -> Tuple[List[int], int, bool, 'info']:
        self.max_steps_per_episode -= 1
        actions = self.possible_actions[action]
        self.swap(actions)
        # Rewards 
        reward = -1
        if self.is_executable_state():
            if actions == (0,0): reward = 0
            # remove the exicutable slice and add a new random slice at the tail
            self.update_state()
            self.max_layers -= 1
            # we are not done except if this was the last layer we can work on this episode
            #if self.max_layers <= 0: done = True
            #else: done = False
            done = self.max_layers <= 0
        elif self.max_steps_per_episode <= 0:
            done = True
            reward = -400
        else: done = False
        info = {}
        
        return self.state, reward, done, info
        

    def render(self):
        pass
    

    def reset(self) -> List[int]:
        self.state = self.make_state()
        self.max_layers = self.depth_of_code
        self.max_steps_per_episode = 200
        return self.state


# swap is given actions which is a tuple of actions or a action, where every action is a tuple with the values
# of two qubits (x, y) whos values should be swaped. x and y are ints between 0 and 8 corresponding to 
# the following qubit notation:
#         [[0, 1, 2],
#          [3, 4, 5],
#          [6, 7, 8]]

# ex. of a tuple of actions: ((0, 3), (4, 5), (7, 8))
# ex. of a single action: (0, 1)

# in the case of a single action we make a list out of it so it's iterable to minimize code
    def swap(self, actions: Tuple[Action]) -> None:
        state = self.state.reshape((self.depth_of_code, self.rows, self.cols))
        if not isinstance(actions[0], tuple):
            actions = [actions]
        for action in actions:
            pos0, pos1 = action

            col0 = pos0 % self.cols
            row0 = int((pos0-col0) / self.cols)  
            col1 = pos1 % self.cols
            row1 = int((pos1-col1) / self.cols)

            for i in range(self.depth_of_code):
                state[i][row0][col0], state[i][row1][col1] = state[i][row1][col1], state[i][row0][col0]
        self.state = state.reshape(self.depth_of_code * self.rows*self.cols)

# get_neighbors returns a list of the qubit notations of all neighbors to a specific qubit. 
# I.e. qubits above, below, right and left of the specific qubit.
    def get_neighbors(self, state: Matrix, row_number: int, column_number: int) -> List[int]:
        a = [[state[i][j] if  i >= 0 and i < len(state) and j >= 0 and j < len(state[0]) else -1
                        for j in range(column_number-1, column_number+2)]
                        for i in range(row_number-1, row_number+2)]
        return [a[0][1], a[1][0], a[1][2], a[2][1]]


#                         [[1,0,0], [[1,0,0],   [[1,0,0],         [[1,0,0],  
# Takes in a state like [  [1,0,2],  [1,0,2], ,  [1,0,2], , ... ,  [1,0,2], ] and checks if all the pairs of 
#                          [2,0,0]]  [2,0,0]]    [2,0,0]]          [2,0,0]] 
# numbers in the first slice are neighbors and if so returns True else returns False

    def is_executable_state(self) -> bool:
        state = self.state.reshape((self.depth_of_code, self.rows, self.cols))
        for row in range(len(state[0])):
            for col in range(len(state[0][0])):
                if state[0][row][col] > 0:
                    if not state[0][row][col] in self.get_neighbors(state[0], row, col):
                        return False
        return True

# We use this once to get all the different swap combinations. I.e. all acceptable combinations of one to four
# swaps. This are the different actions we cound make in one timestep.
    
    def get_possible_actions_sub(self, state: Matrix, used: List[int], max_swaps_per_time_step: int) -> List[Action]:
        if max_swaps_per_time_step == 0:
            return np.asarray([])
        
        possible_actions = []
        
        for i in range(len(state)):
            for j in range(len(state[0])):
                
                usedtmp = used.copy()
                
                if not state[i][j] in usedtmp:
                    neighbors = self.get_neighbors(state, i, j)
                    for neighbor in neighbors:
                        if neighbor >= 0 and not (neighbor, state[i][j]) in possible_actions and not neighbor in usedtmp:
                            possible_actions.append((state[i][j], neighbor))
                            usedtmp.append(state[i][j])
                            usedtmp.append(neighbor)
     
                            for action in self.get_possible_actions_sub(state, usedtmp, max_swaps_per_time_step-1):
                                if isinstance(action, tuple):
                                    possible_actions.append([(state[i][j], neighbor), action])
                                elif isinstance(action, list):
                                    action.append((state[i][j], neighbor))
                                    possible_actions.append(action)
        return possible_actions


    def get_possible_actions(self) -> List[Action]:
        state = np.arange(self.rows * self.cols).reshape((self.rows, self.cols))
        
        possible_actions = self.get_possible_actions_sub(state, self.used, self.max_swaps_per_time_step)
        
        possible_actions = list(map(lambda x: tuple(sorted(x)), possible_actions))
        
        possible_actions.append((0, 0))
        
        return possible_actions

# Creates a shuffled Matrix simulating a slice of quantum code with one to max amount 
# of operations per timestep

# Ex1. [[0, 1, 0],
#       [1, 2, 2],
#       [3, 0, 3]]

# Ex2. [[2, 1],
#       [2, 1]]

    def make_state_slice(self) -> List[int]:
        random = np.random.choice([x for x in range(2, self.rows * self.cols+2) if x % 2])
        state_slice = np.ceil(np.arange(1, random)/2)
        state_slice = np.append(state_slice, np.zeros(self.rows * self.cols-random+1, dtype = int))
        np.random.shuffle(state_slice)
        return state_slice.reshape((self.rows, self.cols))

    # Makes a state out of depth_of_code amount of slices
    def make_state(self) -> List[int]:
        state = np.zeros((self.depth_of_code, self.rows, self.cols))
        for i in range(len(state)):
            state[i] = self.make_state_slice()
        return state.reshape(self.depth_of_code * self.rows * self.cols)


    def update_state(self) -> None:
        tmp = np.roll(self.state.reshape((self.depth_of_code, self.rows, self.cols)), -1, axis=0)
        tmp[self.depth_of_code - 1] = self.make_state_slice()
        self.state = tmp.reshape(self.depth_of_code * self.rows * self.cols)
