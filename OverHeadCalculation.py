from MultiSwapEnvironment import swap_environment
from typing import List, Tuple
from gym.envs.registration import register
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from dqn.dqn import DQN
from dqn.evaluation import evaluate_policy
import matplotlib.pyplot as plt

Matrix = List[List[int]]

depth = 10
rows = 2
cols = 2
gates = 10
max_swaps_per_time_step = -1

n_eval_episodes = 100

modelDir = "models/"

modelName = f"DQNModel({depth},{rows},{cols}).zip"

register(
    id="MultiSwapEnvironment-v0",
    entry_point="MultiSwapEnvironment:swap_environment",
    max_episode_steps=200,
)

venv = make_vec_env("MultiSwapEnvironment-v0", n_envs=1, env_kwargs = {"gates": gates, "depth": depth, "rows": rows, "cols": cols, "max_swaps_per_time_step": max_swaps_per_time_step})
model = DQN.load(modelDir + modelName, env=venv)

def pm_to_state_rep(pm: Matrix):
    action_matrix = pm.tolist()
    action_tuples = [] 
    used_nodes = [] 
    for i in range(len(action_matrix)): 
        if i not in used_nodes: 
            idx = action_matrix[i].index(1) 
            used_nodes.append(idx)
            if idx != i:
                action_tuples.append(tuple((i,idx)))
    idx = -1
    return_a = np.zeros(rows*cols)
    for q0, q1 in action_tuples:
        return_a[q0] = idx
        return_a[q1] = idx
        idx -= 1
    return list(return_a.reshape(rows*cols))

code_size = range(20, 200, 20)

cdr = np.array([])
overhead = np.array([])

for d in code_size:
    env = swap_environment(depth, rows, cols, -1, 200, d)

    a = []
    prelength = np.array([])
    postlength = np.array([])
    for i in range(n_eval_episodes):
        a.append([])
        obs = env.reset()
        prelength = np.append(prelength, len(env.processing(env.get_code(), preprocessing=False)))
        done = False
        while not done:
            action, _states = model.predict(obs[None][None], venv,  deterministic=True)
            obs,r,done,_ = env.step(action[0])
            if action != 0:
                a[i].append(pm_to_state_rep(env.possible_actions[action[0]]))
            a[i].append(obs[0].reshape(rows*cols).tolist())
        postlength = np.append(postlength, len(env.processing(a[i], preprocessing=False)))

    print(np.mean(prelength))
    overhead = np.append(overhead, np.mean(postlength)-np.mean(prelength))
    cdr = np.append(cdr, np.mean(postlength)/np.mean(prelength))

fig, ax = plt.subplots(2)

ax[0].plot(code_size, cdr, color='b')
ax[0].scatter(code_size, cdr, color='b', marker='<')
ax[0].set_ylabel('CDR')

ax[1].plot(code_size, overhead, color='r')
ax[1].scatter(code_size, overhead, color='r', marker='s')
ax[1].set_ylabel('CDO')
ax[1].set_xlabel('Gates')

plt.savefig('CDR-CDO.png', dpi=400, bbox_inches='tight')
plt.show()
