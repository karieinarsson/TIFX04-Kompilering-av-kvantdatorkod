from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math
from typing import List, Tuple
import copy
from stable_baselines3.common.env_checker import check_env
import pygame
import math
# types
Matrix = List[List[int]]
Action = List[int]

#pygame constants
PG_WIDTH  = 100
PG_HEIGHT = 100
X_START   = PG_WIDTH*0.6
Y_START   = PG_HEIGHT*0.6
#Colors
WHITE   = (255,255,255)
BLACK   = (0,0,0)
BLUE    = (0,0,255)
GREEN   = (0,255,0)
RED     = (255,0,0)
CYAN    = (0,255,255)
PURPLE  = (255,0,255)
YELLOW  = (255,255,0)
BROWN   = (165,42,42)
PINK    = (255,20,147)
GREY    = (50.2,50.2,50.2)
PURPLE  = (50.2,0,50.2)
LIME    = (191,255,0)

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
        self.max_layers = depth_of_code + 10
        self.max_episode_steps = 200
        #array of possible actions
        self.possible_actions = self.get_possible_actions()
        #Number of actions we can take
        self.action_space = Discrete(len(self.possible_actions))
        self.observation_space = Box(low=0, high=np.floor(self.rows * self.cols / 2),
                                shape=(1, depth_of_code, rows, cols, ), dtype=np.uint8)
        #The start state
        self.state = self.make_state()
        #max amount of layers per episode
        self.max_layers = self.depth_of_code
        
        #pygame screen initialization
        self.screen = None
        self.isopen = True
    
    def step(self, action: Discrete) -> Tuple[List[int], int, bool, 'info']:
        self.state = self.state.reshape((self.depth_of_code, self.rows*self.cols))
        self.max_episode_steps -= 1
        swap_matrix = self.possible_actions[action]
        self.state = np.matmul(self.state, swap_matrix)
        # Rewards
        reward = self.reward_func(self.state)

        if reward == -1:
            if action == 0: reward = 0
            # remove the exicutable slice and add a new random slice at the tail
            self.state = np.roll(self.state, -1, axis=0)
            self.state[self.depth_of_code - 1] = self.make_state_slice()
            # we are not done except if this was the last layer we can work on this episode
            self.max_layers -= 1
        if self.max_episode_steps <= 0 or self.max_layers <= 0:
            done = True
        else:
            done = False

        info = {}


        self.state = self.state.reshape((self.depth_of_code, self.rows, self.cols))
        return self.state, reward, done, info

    def render(self, mode = "human", render_list = None): 
        if render_list is None:
            return 
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((PG_WIDTH*self.cols,PG_HEIGHT*self.rows))
       
        num_font = pygame.font.SysFont(None,25)
        img0 = num_font.render('0',True,RED)
        img1 = num_font.render('1',True,RED)
        img2 = num_font.render('2',True,RED)
        img3 = num_font.render('3',True,RED)
        img4 = num_font.render('4',True,RED)
        img5 = num_font.render('5',True,RED)
        img6 = num_font.render('6',True,RED)
        img7 = num_font.render('7',True,RED)
        img8 = num_font.render('8',True,RED)
        img9 = num_font.render('9',True,RED) 
        s_img = num_font.render('S',True,BLACK)
        
        dict={
            0:BLACK,
            1:GREEN,
            2:BLUE,
            3:PURPLE,
            4:YELLOW,
            5:BROWN,
            6:PINK,
            7:GREY,
            8:PURPLE,
            9:LIME
            }

        num_dict={
                0:img0,
                1:img1,
                2:img2,
                3:img3,            
                4:img4,            
                5:img5,
                6:img6,
                7:img7,
                8:img8,
                9:img9  
                }
        
        num_matrix = []                
        tmp = 0            
        for _ in range(self.rows):     
            tmpm = []              
            for _ in range(self.cols): 
                tmpm.append(tmp)       
                tmp += 1               
            num_matrix.append(tmpm) 


        surface = pygame.Surface(self.screen.get_size())

        pygame.draw.rect(surface,WHITE,surface.get_rect())
         
        #row / col %

        for j in range(1,self.cols+1):
            for i in range(1,self.rows+1):
                pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                if j < self.rows:
                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                if i < self.cols: 
                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                pygame.draw.circle(surface,dict.get(render_list[0][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))

        self.screen.blit(surface,(0,0))
        pygame.display.flip()

        index = 0
        running = True

        while running:
            ev = pygame.event.get()

            for event in ev:

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if index%2 == 0:                    
                        pygame.draw.rect(surface,WHITE,surface.get_rect())
                        for j in range(1,self.cols+1):
                            for i in range(1,self.rows+1):
                                pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                                if j < self.rows:
                                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                                if i < self.cols: 
                                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                                    surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))

                    if event.key == pygame.K_n:
                        #next one
                        if index == len(render_list)-1:
                            print("At last obs")
                        else:
                            index += 1
                        
                        if type(render_list[index]) is list:
                            pygame.draw.rect(surface,WHITE,surface.get_rect())

                            for j in range(1,self.cols+1):
                                for i in range(1,self.rows+1):
                                    pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                                    if j < self.rows:
                                        pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                                    if i < self.cols: 
                                        pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                                    pygame.draw.circle(surface,dict.get(render_list[index][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                    surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))

                            
                       
                        else:
                            for j in range(1,self.cols+1):
                                for i in range(1,self.rows+1):
                                    pygame.draw.circle(surface,dict.get(render_list[index-1][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                    surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))
                            swap_matrix = self.possible_actions[render_list[index]]
                            tuple_list = self.action_render(swap_matrix)

                            for t in tuple_list:
                                r0 = math.floor(t[0]/self.cols)
                                c0 = t[0]%self.cols
                                r1 = math.floor(t[1]/self.cols)
                                c1 = t[1]%self.cols
                                x0 = X_START*(c0+1)
                                y0 = Y_START*(r0+1)
                                x1 = X_START*(c1+1)
                                y1 = Y_START*(r1+1)
                                x = x1+((x0-x1)/2)
                                y = y1+((y0-y1)/2)
                                pygame.draw.rect(surface,CYAN,pygame.Rect((x-10,y-10),(20,20)))
                                surface.blit(s_img,(x-6,y-8))
                            
                            num_matrix_tmp = np.matmul(np.asarray(num_matrix).reshape(self.rows*self.cols),swap_matrix)
                            num_matrix = num_matrix_tmp.reshape((self.rows,self.cols)).tolist()

                        self.screen.blit(surface,(0,0))
                        pygame.display.flip()

                    if event.key == pygame.K_b:
                        #back one
                        if index == 0:
                            print("At first obs")
                        else:    
                            index -= 1
                        
                        if type(render_list[index]) is list:
                            pygame.draw.rect(surface,WHITE,surface.get_rect())
                            for j in range(1,self.cols+1):
                                for i in range(1,self.rows+1):
                                    pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                                    if j < self.rows:
                                        pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                                    if i < self.cols: 
                                        pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                                    pygame.draw.circle(surface,dict.get(render_list[index][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                    surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))
                        
                        else: 
                            num_matrix_tmp = np.matmul(np.asarray(num_matrix).reshape(self.rows*self.cols),swap_matrix.T)
                            num_matrix = num_matrix_tmp.reshape((self.rows,self.cols)).tolist()
                            for i in range(len(num_matrix)): 
                                num_matrix[i] = list(map(int,num_matrix[i]))

                            for j in range(1,self.cols+1):
                                for i in range(1,self.rows+1):
                                    pygame.draw.circle(surface,dict.get(render_list[index-1][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                    surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))
                            swap_matrix = self.possible_actions[render_list[index]]
                            tuple_list = self.action_render(swap_matrix)
                            
                            for t in tuple_list:
                                r0 = math.floor(t[0]/self.cols)
                                c0 = t[0]%self.cols
                                r1 = math.floor(t[1]/self.cols)
                                c1 = t[1]%self.cols
                                x0 = X_START*(c0+1)
                                y0 = Y_START*(r0+1)
                                x1 = X_START*(c1+1)
                                y1 = Y_START*(r1+1)
                                x = x1+((x0-x1)/2)
                                y = y1+((y0-y1)/2)
                                pygame.draw.rect(surface,CYAN,pygame.Rect((x-10,y-10),(20,20)))
                                surface.blit(s_img,(x-6,y-8))
                           


                        self.screen.blit(surface,(0,0))
                        pygame.display.flip()



        self.screen.blit(surface,(0,0))

        pygame.event.pump()
        pygame.display.flip()
        return self.isopen

    def reset(self) -> List[int]:
        self.state = self.make_state()
        self.max_layers = self.depth_of_code
        self.max_episode_steps = 200
        return self.state

#                                             [[1,0,0], [[1,0,0],   [[1,0,0],         [[1,0,0],  
# is_exicutable_state takes in a state like [  [1,0,2],  [1,0,2], ,  [1,0,2], , ... ,  [1,0,2], ]  
#                                              [2,0,0]]  [2,0,0]]    [2,0,0]]          [2,0,0]] 
# and checks if all the pairs of numbers in the first slice are neighbors and 
#if so returns True else returns False

    def is_executable_state(self, state) -> bool:
        for pos in range(self.rows * self.cols):
            gate = state[0][pos]
            if gate > 0:
                neighbors = [state[0][pos+i] if pos+i >= 0 and pos+i < self.rows*self.cols 
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
            return_possible_actions = np.zeros((len(possible_actions)+1, self.rows*self.cols, self.rows*self.cols))
            return_possible_actions[0] = np.identity(self.rows*self.cols)
            for idx, action in enumerate(possible_actions):
                m = np.identity(self.rows*self.cols)
                for swap in action:
                    pos1, pos2 = swap
                    m[pos1][pos1] = 0
                    m[pos2][pos2] = 0
                    m[pos1][pos2] = 1
                    m[pos2][pos1] = 1
                return_possible_actions[idx+1] = m
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
            state_slice[i-1] = i
            state_slice[i-1+max_gates] = i
        np.random.shuffle(state_slice)
        return state_slice

    # Makes a state out of depth_of_code amount of slices
    def make_state(self) -> List[int]:
        state = np.zeros((self.depth_of_code, self.rows, self.cols))
        for i in range(len(state)):
            state[i] = self.make_state_slice().reshape((self.rows, self.cols))
        return state

    def reward_func(self, state) -> int:
        if self.is_executable_state(state):
            return -1
        return -2


    def action_render(self,action_matrix):                                                                                                                                                
         action_matrix = action_matrix.tolist()
         action_tuples = [] 
         used_nodes = [] 
         for i in range(len(action_matrix)): 
             if i not in used_nodes: 
                 idx = action_matrix[i].index(1) 
                 used_nodes.append(idx)
                 if idx != i:
                     action_tuples.append(tuple((i,idx)))
         return action_tuples

if __name__ == '__main__':
    main()
