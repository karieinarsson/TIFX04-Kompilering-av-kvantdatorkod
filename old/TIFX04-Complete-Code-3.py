from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    depthOfCode = 10
    rows = 2
    cols = 2  

    # Create environment
    env = Kvant(depthOfCode, rows, cols)

    # Instantiate the agent
    model = DQN('MlpPolicy', env, verbose=1)#, exploration_final_eps = 0.1)

    # Train the agent

    model.learn(total_timesteps = int(1e5))

    # Save the agent

    modelName = "KvantShit("+ str(env.depthOfCode) + ", " + str(env.rows) + ", " + str(env.cols) + ")"

    model.save(modelName)

    # delete trained model to demonstrate loading
    del model  

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = DQN.load(modelName, env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    print(mean_reward)









#Our enviorment
class Kvant(Env):

# swap is given actions which is a tuple of actions or a action, where every action is a tuple with the values
# of two qubits (x, y) whos values should be swaped. x and y are ints between 0 and 8 corresponding to 
# the following qubit notation:
#         [[0, 1, 2],
#          [3, 4, 5],
#          [6, 7, 8]]

# ex. of a tuple of actions: ((0, 3), (4, 5), (7, 8))

# ex. of a single action: (0, 1)

# in the case of a single action we make a list out of it so it's iterable to minimize code
    def swap(self, actions):
        state = self.state.reshape((self.depthOfCode, self.rows, self.cols))
        if type(actions[0]) != tuple:
            actions = [actions]
        for action in actions:
            pos0, pos1 = action

            col0 = pos0 % self.cols
            row0 = int((pos0-col0)/self.cols)  
            col1 = pos1 % self.cols
            row1 = int((pos1-col1)/self.cols)

            for i in range(self.depthOfCode):
                state[i][row0][col0], state[i][row1][col1] = state[i][row1][col1], state[i][row0][col0]
                #self.state[pos0+i*self.cols], self.state[pos1+i*self.cols] = self.state[pos1+i*self.cols], self.state[pos0+i*self.cols]
        self.state = state.reshape(self.depthOfCode*self.rows*self.cols)

# getNeighbors returns a list of the qubit notations of all neighbors to a specific qubit. 
# I.e. qubits above, below, right and left of the specific qubit.

    def getNeighbors(self, state, row_number, column_number):
        a = [[state[i][j] if  i >= 0 and i < len(state) and j >= 0 and j < len(state[0]) else -1
                        for j in range(column_number-1, column_number+2)]
                            for i in range(row_number-1, row_number+2)]
        return [a[0][1], a[1][0], a[1][2], a[2][1]]


#                         [[1,0,0], [[1,0,0],   [[1,0,0],         [[1,0,0],  
# Takes in a state like [  [1,0,2],  [1,0,2], ,  [1,0,2], , ... ,  [1,0,2], ] and checks if all the pairs of 
#                          [2,0,0]]  [2,0,0]]    [2,0,0]]          [2,0,0]] 
# numbers in the first slice are neighbors and if so returns True else returns False

    def isExecutableState(self):
        state = self.state.reshape((self.depthOfCode, self.rows, self.cols))
        for row in range(len(state[0])):
            for col in range(len(state[0][0])):
                if state[0][row][col] > 0:
                    if not state[0][row][col] in self.getNeighbors(state[0], row, col):
                        return False

        return True

# We use this once to get all the different swap combinations. I.e. all acceptable combinations of one to four
# swaps. This are the different actions we cound make in one timestep.
    
    def getPossibleActionsSub(self, state, used, maxSwapsPerTimeStep):
        if maxSwapsPerTimeStep == 0:
            return np.asarray([])
        
        possibleActions = []
        
        for i in range(len(state)):
            for j in range(len(state[0])):
                
                usedtmp = used.copy()
                
                if not state[i][j] in usedtmp:
                    neighbors = self.getNeighbors(state, i, j)
                    for neighbor in neighbors:
                        if neighbor >= 0 and not (neighbor, state[i][j]) in possibleActions and not neighbor in usedtmp:
                            possibleActions.append((state[i][j], neighbor))
                            usedtmp.append(state[i][j])
                            usedtmp.append(neighbor)
     
                            for action in self.getPossibleActionsSub(state, usedtmp, maxSwapsPerTimeStep-1):
                                if type(action) == tuple:
                                    possibleActions.append([(state[i][j], neighbor), action])
                                elif type(action) == list:
                                    action.append((state[i][j], neighbor))
                                    possibleActions.append(action)
                                    
            
        return possibleActions

    def getPossibleActions(self):
        state = np.arange(self.rows*self.cols).reshape((self.rows,self.cols))
        
        possibleActions = self.getPossibleActionsSub(state, self.used, self.maxSwapsPerTimeStep)
        
        possibleActions = list(map(lambda x: tuple(sorted(x)), possibleActions ))
        
        possibleActions.append((0, 0))
        
        return possibleActions
        

# Creates a shuffled Matrix simulatinga slice of quantum code with one to max amount 
# of operations per timestep

# Ex1. [[0, 1, 0],
#       [1, 2, 2],
#       [3, 0, 3]]

# Ex2. [[2, 1],
#       [2, 1]]

    def makeStateSlice(self):
        random = np.random.choice([x for x in range(2, self.rows*self.cols+2) if x % 2])
        stateSlice = np.ceil(np.arange(1, random)/2)
        stateSlice = np.append(stateSlice, np.zeros(self.rows*self.cols-random+1))
        np.random.shuffle(stateSlice)
        return stateSlice.reshape((self.rows,self.cols))

    # Makes a state out of depthOfCode amount of slices
    def makeState(self):
        state = np.zeros((self.depthOfCode,self.rows,self.cols))
        for i in range(len(state)):
            state[i] = self.makeStateSlice()
        return state.reshape(self.depthOfCode*self.rows*self.cols)


    def updateState(self):
        tmp = np.roll(self.state.reshape((self.depthOfCode, self.rows, self.cols)), -1, axis = 0)
        tmp[self.depthOfCode - 1] = self.makeStateSlice()
        self.state = tmp.reshape(self.depthOfCode*self.rows*self.cols)


    def __init__(self, depthOfCode, rows, cols, used = [], 
                                maxSwapsPerTimeStep = -1):
        
        self.depthOfCode = depthOfCode
        self.rows = rows
        self.cols = cols
        self.used = used
        if maxSwapsPerTimeStep < 0 or maxSwapsPerTimeStep > np.floor(self.rows*self.cols/2):
            self.maxSwapsPerTimeStep = np.floor(self.rows*self.cols/2)
        else:
            self.maxSwapsPerTimeStep = maxSwapsPerTimeStep

        self.maxStepsPerEpisode = 200

        #array of possible actions
        self.possibleActions = self.getPossibleActions()

        #Number of actions we can take
        self.action_space = Discrete(len(self.possibleActions))
         
        #
        self.observation_space = Box(low=0, high=math.floor(rows*cols/2),
                                shape=(depthOfCode*rows*cols, ), dtype=np.int)
        
        #The start state
        self.state = self.makeState()
        
        #max amount of layers per episode
        self.maxLayers = self.depthOfCode
        
    def step(self, action):
        self.maxStepsPerEpisode -= 1

        actions = self.possibleActions[action]

        self.swap(actions)
         
        
        # Rewards 
        reward = -1
        
        if self.isExecutableState():
            if actions == (0,0):
                reward = 0
            
            # remove the exicutable slice and add a new random slice at the tail
            
            self.updateState()
            
            self.maxLayers -= 1
            
            # we are not done except if this was the last layer we can work on this episode
            if self.maxLayers <= 0:
                done = True
            else:
                done = False
        
        elif self.maxStepsPerEpisode <= 0:
            done = True
            reward = -400

        else:
            done = False
        
        info = {}
        
        return self.state, reward, done, info
        
    def render(self):
        pass
    
    def reset(self):
        self.state = self.makeState()
        self.maxLayers = self.depthOfCode
        self.maxStepsPerEpisode = 200
        return self.state


if __name__ == "__main__":
    main()
