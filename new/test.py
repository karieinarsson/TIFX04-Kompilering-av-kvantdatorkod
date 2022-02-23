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
            neighbors = [state[i] if i >= 0 and i < rows*cols else 0 
                    for i in [pos+1, pos-1, pos+rows, pos-rows]]
            print(gate, neighbors)
            if not gate in neighbors:
                return False
    return True

def swap(state, actions):
    if not isinstance(actions[0], tuple):
        actions = [actions]
    for action in actions:
        pos0, pos1 = action
        for i in range(10):
            state[pos0+i*3*3], state[pos1+i*3*3] = state[pos1+i*3*3], state[pos0+i*3*3]

    return state


def get_possible_actions(m, rows, cols, iterations, used):
    possible_actions = []
    for pos in m:
        if not pos in used:
            neighbors = [m[i] if i >= 0 and i < rows*cols and not m[i] in used else -1 
                                    for i in [pos+1, pos-1, pos+rows, pos-rows]]
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
                            for action in get_possible_actions(m, rows, cols, iterations-1, used_tmp):
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

print(get_state(3,3,3))
