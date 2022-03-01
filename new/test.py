import numpy as np
import copy
import math

a = np.arange(9).reshape((3,3))

b = np.zeros((10, 3, 3))
for i in range(10):
    b[i] = copy.deepcopy(a)

c = b.reshape(90)

actions = ((0,1),(2,5),(3,4),(8,7))

def isExecutableState(state, rows, cols):
    for pos in range(rows * cols):
        gate = state[pos]
        if gate > 0:
            neighbors = [state[pos+i] if pos+i >= 0 and pos+i < rows*cols
                    and not (pos%rows == 0 and i == -1) 
                    and not (pos%rows == rows-1 and i == 1) else 0 
                    for i in [1, -1, rows, -rows]]
            if not gate in neighbors:
                 False
    return True

def swap(state, actions):
    if not isinstance(actions[0], tuple):
        actions = [actions]
    for action in actions:
        pos0, pos1 = action
        for i in range(10):
            state[pos0+i*3*3], state[pos1+i*3*3] = state[pos1+i*3*3], state[pos0+i*3*3]

    return state


def get_possible_actions(rows, cols, iterations = None, used = None):
    if iterations is None:
        iterations = math.floor(rows*cols/2)
    if used is None:
        used = []
    possible_actions = []
    m = np.arange(rows*cols)
    for pos in m:
        if not pos in used:
            neighbors = [m[pos+i] if pos+i >= 0 and pos+i < rows*cols
                    and not m[pos+i] in used
                    and not (pos%rows == 0 and i == -1) 
                    and not (pos%rows == rows-1 and i == 1) else -1 
                    for i in [1, -1, rows, -rows]]
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
                            for action in get_possible_actions(rows, cols, iterations-1, used_tmp):
                                action.append(a)
                                action.sort()
                                if not action in possible_actions:
                                    possible_actions.append(action)

    return possible_actions


def get_state_slice(rows, cols):
    max_gates = math.floor(rows*cols/2)
    state_slice = np.zeros(rows*cols)
    for i in range(1, np.random.choice(range(2, max_gates+2))):
        state_slice[i] = i
        state_slice[i+max_gates] = i
    np.random.shuffle(state_slice)
    return state_slice

def get_state(rows, cols, depth_of_code):
    state = np.zeros((depth_of_code, rows*cols))
    for i in range(depth_of_code):
        state[i] = get_state_slice(rows, cols)
    return state.reshape(depth_of_code*rows*cols)

def num_permutations_of_executable_states(rows, cols):
    states = []
    o = range(1, math.floor(rows*cols/2)+1)
    for gate in o:
        if gate == 1:
            a = [[ 0 for l in range(rows*cols)]]
        else:
            a = copy.deepcopy(states)
        for s in a:
            #for gate in range(1, gates+1):
            for i in range(rows*cols):
                if s[i] == 0:
                    b = s.copy()
                    b[i] = gate
                    neighbor_pos =  [(b[i+j], i+j) if i+j >= 0 and i+j < rows*cols 
                            and b[i+j] == 0 and not (i%rows == 0 and j == -1) 
                            and not (i%rows == rows-1 and j == 1) else (-1, -1) 
                            for j in [1, -1, rows, -rows]]
                    for n in neighbor_pos:
                        val, j = n
                        if val != -1:
                            c = b.copy()
                            c[j] = gate
                            states.append(c)
    return_states = []
    for i in range(len(states)):
        if not states[i] in return_states:
            return_states.append(states[i])


    return return_states

def render(actions, state, rows, cols, depth_of_code):
    print_string = ''
    for _ in range((cols*2+2)*depth_of_code+1):
        print_string += '-'
    print(print_string)
    for r in range(rows*2-1):
        print_string = ''
        if r%2 == 0:
            row = int(r/2)
            for depth in range(depth_of_code):
                print_string += '|'
                for col in range(cols):
                    if col == 0:
                        print_string += ' '
                    else:
                        if [(col-1) + row*cols, col + row*cols] in actions:
                            print_string += '-'
                        else:
                            print_string += ' '
                    
                    print_string += str(int(state[col + row*cols + depth*rows*cols]))

                    if col == cols-1:
                        print_string += ' '
                if depth == depth_of_code-1:
                    print_string += '|'
        else:
            for depth in range(depth_of_code):
                print_string += '| '
                for col in range(cols):
                    if [col + int((r-1)/2)*cols, col + int((r+1)/2)*cols] in actions:     
                        print_string += '|'                                 
                    else:                                                   
                        print_string += ' ' 
                    
                    if col != cols-1:
                        print_string += ' '
                print_string += ' '
                if depth == depth_of_code-1:
                    print_string += '|'

        print(print_string) 
    print_string = ''
    for _ in range((cols*2+2)*depth_of_code+1):
        print_string += '-'
    print(print_string)


rows = 3
cols = 2
depth_of_code = 10

state = get_state(rows, cols, depth_of_code)
actions = [[0,1]]
render(actions, state, rows, cols, depth_of_code)
