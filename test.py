from MultiSwapEnviorment import swap_enviorment
import numpy as np

env = swap_enviorment(5,2,3,-1)

pa = env.possible_actions

obs = np.arange(6)

a = np.matmul(obs, pa)

print(a)
