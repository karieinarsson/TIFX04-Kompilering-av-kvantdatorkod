from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math
from typing import List, Tuple
import copy
from stable_baselines3.common.env_checker import check_env

# types
Matrix = List[List[int]]
Action = List[int]

def main():
    env = swap_enviorment(10,3,3)
    check_env(env)
    

#Our enviorment
class swap_enviorment(Env):
    def __init__(self, depth_of_code: int, rows: int, cols: int, 
                 max_swaps_per_time_step: int=-1, used: List[int]=None) -> None:
        self.depth_of_code = depth_of_code
        self.rows = rows
        self.cols = cols
        if used is None:
            used = []
        self.used = used
        if max_swaps_per_time_step < 0 or max_swaps_per_time_step > np.floor(self.rows*self.cols/2):
            self.max_swaps_per_time_step = np.floor(self.rows * self.cols/2)
        else: 
            self.max_swaps_per_time_step = max_swaps_per_time_step
        self.max_steps_per_episode = 200
        #array of possible actions
        self.possible_actions = self.get_possible_actions()
        #Number of actions we can take
        self.action_space = Discrete(len(self.possible_actions))
        print(self.action_space)
        self.observation_space = Box(low=0, high=math.floor(rows * cols/2),
                                shape=(depth_of_code, rows, cols, ), dtype=np.uint8)
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
        

    def render(self, mode = "human"): 
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
    def swap(self, action: Matrix) -> None:
        self.state = self.state.reshape((self.depth_of_code, self.rows*self.cols))
        self.state = np.matmul(self.state, action)
        self.state = self.state.reshape((self.depth_of_code, self.rows, self.cols))

#                                             [[1,0,0], [[1,0,0],   [[1,0,0],         [[1,0,0],  
# is_exicutable_state takes in a state like [  [1,0,2],  [1,0,2], ,  [1,0,2], , ... ,  [1,0,2], ]  
#                                              [2,0,0]]  [2,0,0]]    [2,0,0]]          [2,0,0]] 
# and checks if all the pairs of numbers in the first slice are neighbors and 
#if so returns True else returns False

    def is_executable_state(self) -> bool:
        state = self.state.reshape(self.depth_of_code*self.rows*self.cols)
        for pos in range(self.rows * self.cols):
                gate = state[pos]
                if gate > 0:
                    neighbors = [state[pos+i] if pos+i >= 0 and pos+i < self.rows*self.cols 
                            and not (pos%self.rows == 0 and i == -1) 
                            and not (pos%self.rows == self.rows-1 and i == 1) else 0 
                            for i in [1, -1, self.rows, -self.rows]]
                    if not gate in neighbors:
                        return False
        return True

# We use this once to get all the different swap combinations. I.e. all acceptable combinations of one to four
# swaps. This are the different actions we cound make in one timestep.
    
    def get_possible_actions(self, iterations = None, used = None):
        if used is None:
            used = []
        if iterations is None or iterations == -1:
            iterations = self.max_swaps_per_time_step
        m = np.arange(self.rows*self.cols)
        possible_actions = []
        for pos in m:
            if not pos in used:
                neighbors = [m[pos+i] if pos+i >= 0 and pos+i < self.rows*self.cols 
                        and not m[pos+i] in used
                        and not (pos%self.rows == 0 and i == -1) 
                        and not (pos%self.rows == self.rows-1 and i == 1) else -1 
                        for i in [1, -1, self.rows, -self.rows]]
                for target in neighbors:
                    if target != -1:
                        a = [pos, target]
                        a.sort()
                        if not [a] in possible_actions:
                            used_tmp = used.copy()
                            possible_actions.append([a])
                            used_tmp.append(pos)
                            used_tmp.append(target)
                            if iterations >= 1: 
                                for action in self.get_possible_actions(iterations = iterations-1, used = used_tmp):
                                    action.append(a)
                                    action.sort()
                                    if not action in possible_actions:
                                        possible_actions.append(action)

        if iterations == self.max_swaps_per_time_step:
            return_possible_actions = []
            for action in possible_actions:
                if action != [[0,0]]:
                    m = np.identity(self.rows*self.cols)
                    for swap in action:
                        pos1, pos2 = swap
                        m[pos1][pos1] = 0
                        m[pos2][pos2] = 0
                        m[pos1][pos2] = 1
                        m[pos2][pos1] = 1
                    return_possible_actions.append(m)
            return_possible_actions.append(np.identity(self.rows*self.cols))
            return return_possible_actions
        
        return possible_actions

# Creates a shuffled Matrix simulating a slice of quantum code with one to max amount 
# of operations per timestep

# Ex1. [[0, 1, 0],
#       [1, 2, 2],
#       [3, 0, 3]]

# Ex2. [[2, 1],
#       [2, 1]]

    def make_state_slice(self):
        max_gates = math.floor(self.rows*self.cols/2)
        state_slice = np.zeros(self.rows*self.cols)
        for i in range(1, np.random.choice(range(2, max_gates+2))):
            state_slice[i] = i
            state_slice[i+max_gates] = i
        np.random.shuffle(state_slice)
        return state_slice.reshape((3,3))

    # Makes a state out of depth_of_code amount of slices
    def make_state(self) -> List[int]:
        state = np.zeros((self.depth_of_code, self.rows, self.cols))
        for i in range(len(state)):
            state[i] = self.make_state_slice()
        return state.reshape((self.depth_of_code, self.rows, self.cols))


    def update_state(self) -> None:
        self.state = np.roll(self.state, -1, axis=0)
        self.state[self.depth_of_code - 1] = self.make_state_slice()

if __name__ == '__main__':
    main()
