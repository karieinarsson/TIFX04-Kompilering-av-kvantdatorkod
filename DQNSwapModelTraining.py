from MultiSwapEnviorment import swap_enviorment

import numpy as np
import os
import matplotlib.pyplot as plt

from dqn.vec_env import DummyVecEnv

from gym.envs.registration import register

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from dqn.env_util import make_vec_env
from dqn.swap_vec_env import SwapVecEnv
from dqn.dqn import DQN

#env variables
depth_of_code = 10
rows = 3
cols = 2
max_swaps_per_time_step = -1

#model variables
learning_starts = 0 #int(1e4)
verbose = 1
exploration_fraction = 0.2
exploration_initial_eps = 1
exploration_final_eps = 0.1
batch_size = 128
learning_rate = 0.001
target_update_interval = 10000
tau = 0.2
gamma = 0.99
train_freq = 4

#training variables
total_timesteps = int(1e5)
log_interval = 2

register(
    id="MultiSwapEnviorment-v0",
    entry_point="MultiSwapEnviorment:swap_enviorment",
    max_episode_steps=200,
)

venv = make_vec_env("MultiSwapEnviorment-v0", env_kwargs = {"depth_of_code": depth_of_code, "rows": rows, "cols": cols, "max_swaps_per_time_step": max_swaps_per_time_step})

# Intantiate the agent
model = DQN('CnnPolicy', 
            venv, 
            verbose = verbose,
            train_freq = train_freq,
            gamma = gamma,
            tau = tau,
            target_update_interval = target_update_interval,
            learning_starts = learning_starts, 
            exploration_fraction = exploration_fraction, 
            exploration_final_eps = exploration_final_eps, 
            exploration_initial_eps = exploration_initial_eps,
            batch_size = batch_size,
            optimize_memory_usage = True,
            learning_rate = learning_rate
        )

# Train the agent
model.learn(total_timesteps = total_timesteps, log_interval = log_interval)

# Save the agent
#model_dir = "models/"
#model_name = "DQNModel("+ str(env.depth_of_code) + "," + str(env.rows) + "," + str(env.cols) + ',' + str(env.max_swaps_per_time_step) + ")"
#model.save(model_dir + model_name)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print(mean_reward)

