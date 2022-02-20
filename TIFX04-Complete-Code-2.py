#!/usr/bin/env python
# coding: utf-8

# # Pip install all needed packages

# In[1]:


#get_ipython().system('pip install gym')
#get_ipython().system('pip install stable_baselines3')
#get_ipython().system('pip install numpy')


# # Imports

# In[2]:

import numpy as np
import math
import sys

from gym import Env
from gym.spaces import Discrete, Box

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# # Global variables

# In[3]:

depthOfCode = 5
rows = 2
cols = 2
usedQubits = []
maxSwapsPerTimeStep = math.floor(rows*cols/2)

# # Functions

# In[4]:


# swap is given actions which is a tuple of actions or a action, where every action is a tuple with the values
# of two qubits (x, y) whos values should be swaped. x and y are ints between 0 and 8 corresponding to 
# the following qubit notation:
#         [[0, 1, 2],
#          [3, 4, 5],
#          [6, 7, 8]]

# ex. of a tuple of actions: ((0, 3), (4, 5), (7, 8))

# ex. of a single action: (0, 1)

# in the case of a single action we make a list out of it so it's iterable to minimize code
def swap(state, actions):
    if type(actions[0]) != tuple:
        actions = [actions]
    for action in actions:
        pos0, pos1 = action
        
        col0 = pos0%cols
        row0 = int((pos0-col0)/cols)  
        col1 = pos1%cols
        row1 = int((pos1-col1)/cols)
        
        for i in range(len(state)):
            state[i][row0][col0], state[i][row1][col1] = state[i][row1][col1], state[i][row0][col0]


# In[5]:


# getNeighbors returns a list of the qubit notations of all neighbors to a specific qubit. 
# I.e. qubits above, below, right and left of the specific qubit.

def getNeighbors(state, row, column):
    a = [[state[i][j] if  i >= 0 and i < len(state) and j >= 0 and j < len(state[0]) else -1
                    for j in range(column-1, column+2)]
                        for i in range(row-1, row+2)]
    return [a[0][1], a[1][0], a[1][2], a[2][1]]


# In[6]:


#                         [[1,0,0], [[1,0,0],   [[1,0,0],         [[1,0,0],  
# Takes in a state like [  [1,0,2],  [1,0,2], ,  [1,0,2], , ... ,  [1,0,2], ] and checks if all the pairs of 
#                          [2,0,0]]  [2,0,0]]    [2,0,0]]          [2,0,0]] 
# numbers in the first slice are neighbors and if so returns True else returns False

def isExecutableState(state):
    for row in range(len(state[0])):
        for col in range(len(state[0][0])):
            if state[0][row][col] > 0:
                if not state[0][row][col] in getNeighbors(state[0], row, col):
                    return False

    return True


# In[7]:


# We use this once to get all the different swap combinations. I.e. all acceptable combinations of one to four
# swaps. This are the different actions we cound make in one timestep.

def getPossibleActions(maxSwapsPerTimeStep):
    state = np.arange(rows*cols).reshape((rows,cols))
    
    possibleActions = getPossibleActionsSub(state, usedQubits, maxSwapsPerTimeStep)
    
    possibleActions = list(map(lambda x: tuple(sorted(x)), possibleActions ))
    
    possibleActions.append((0, 0))
    
    return possibleActions
    
def getPossibleActionsSub(state, used, maxSwapsPerTimeStep):
    if maxSwapsPerTimeStep == 0:
        return np.asarray([])
    
    possibleActions = []
    
    for i in range(len(state)):
        for j in range(len(state[0])):
            
            usedtmp = used.copy()
            
            if not state[i][j] in usedtmp:
                neighbors = getNeighbors(state, i, j)
                for neighbor in neighbors:
                    if neighbor >= 0 and not (neighbor, state[i][j]) in possibleActions and not neighbor in usedtmp:
                        possibleActions.append((state[i][j], neighbor))
                        usedtmp.append(state[i][j])
                        usedtmp.append(neighbor)
 
                        for action in getPossibleActionsSub(state, usedtmp, maxSwapsPerTimeStep-1):
                            if type(action) == tuple:
                                possibleActions.append([(state[i][j], neighbor), action])
                            elif type(action) == list:
                                action.append((state[i][j], neighbor))
                                possibleActions.append(action)
                                
        
    return possibleActions


# In[8]:


# Creates a shuffled Matrix simulatinga slice of quantum code with one to max amount 
# of operations per timestep

# Ex1. [[0, 1, 0],
#       [1, 2, 2],
#       [3, 0, 3]]

# Ex2. [[2, 1],
#       [2, 1]]

def makeStateSlice():
    random = np.random.choice([x for x in range(2, rows*cols+2) if x % 2])
    stateSlice = np.ceil(np.arange(1, random)/2)
    stateSlice = np.append(stateSlice, np.zeros(rows*cols-random+1))
    np.random.shuffle(stateSlice)
    return stateSlice.reshape((rows,cols))


# In[9]:


# Makes a state out of depthOfCode amount of slices
def makeState():
    state = np.zeros((depthOfCode,rows,cols))
    for i in range(len(state)):
        state[i] = makeStateSlice()
    return state


# # Enviotment definition and sub functions

# In[10]:


possibleActions = getPossibleActions( maxSwapsPerTimeStep )


# In[11]:


#Our enviorment
class Kvant(Env):
    def __init__(self):
        #array of possible actions
        self.possibleActions = possibleActions
        
        #self.possibleActions = getPossibleActions(1) #this for only 1 swap at a time
        
        #Number of actions we can take
        self.action_space = Discrete(len(self.possibleActions))
        
        #
        self.observation_space = Box(low=0, high=math.floor(rows*cols/2), shape=(depthOfCode, rows, cols), dtype=np.int)
        
        #The start state
        self.state = makeState()
        
        #max amount of layers per episode
        self.maxLayers = depthOfCode
        
    def step(self, action):
        
        actions = self.possibleActions[action]

        swap(self.state, actions)
         
        
        # Rewards 
        reward = -1
        
        if isExecutableState(self.state):
            if actions == (0,0):
                reward = 0
            
            # remove the exicutable slice and add a new random slice at the tail
            self.state = np.roll(self.state, -1, axis = 0)
            self.state[depthOfCode - 1] = makeStateSlice()
            
            self.maxLayers -= 1
            
            # we are not done except if this was the last layer we can work on this episode
            if self.maxLayers <= 0:
                done = True
            else:
                done = False
            
        else:
            done = False
        
        info = {}
        
        return self.state, reward, done, info
        
    def render(self):
        pass
    
    def reset(self):
        self.state = makeState()
        #self.maxTimeSteps = 5
        self.maxLayers = depthOfCode
        return self.state


# # DQN agent shit

# In[16]:


# Create environment
env = Kvant()

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1, exploration_final_eps = 0.1)
# Train the agent
model.learn(total_timesteps=int(1e6))

# Save the agent
model.save("KvantShit")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("KvantShit", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# In[17]:


print(mean_reward)

