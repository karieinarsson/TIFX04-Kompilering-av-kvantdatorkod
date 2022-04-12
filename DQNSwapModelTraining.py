## Update: Logs rollout, time and train to plot them in tensorboard. 
## Upload tensorboard event file using the command: tensorboard --logdir '.\logdir\DQNModel(StateToValue)_[#nr]\'
## Tensorboard is the opened in browser with the specified adress in terminal. For me this is http://localhost:6006/
## This version also creates a .txt file in the .\logdir\DQNModel(StateToValue)_[#nr] directory with the training input.

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

from dqn.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from dqn.swap_vec_env import SwapVecEnv
from dqn.dqn import DQN

#env variables
depth_of_code = 5
rows = 3
cols = 2
max_swaps_per_time_step = -1

#model variables (previously 2e4)
learning_starts = int(2e4)
verbose = 1
exploration_fraction = 0.5
exploration_initial_eps = 1
exploration_final_eps = 0.05
batch_size = 512
learning_rate = 0.001
target_update_interval = int(2e4)
tau = 0.3
gamma = 0.4
train_freq = 4

#training variables (previously 1e5)
total_timesteps = int(1e5)
log_interval = 4

#evaluation
n_eval_episodes = 20


register(
    id="MultiSwapEnviorment-v0",
    entry_point="MultiSwapEnviorment:swap_enviorment",
    max_episode_steps=200,
)

venv = make_vec_env("MultiSwapEnviorment-v0", n_envs = 7, env_kwargs = {"depth_of_code": depth_of_code, "rows": rows, "cols": cols, "max_swaps_per_time_step": max_swaps_per_time_step})

# Defining agent name
model_dir = "models/"
model_name = "DQNModel(StateToValue)"
logdir="logdir/"

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
            learning_rate = learning_rate,
            tensorboard_log=logdir
        )

# Train the agent
model.learn(total_timesteps = total_timesteps, log_interval = log_interval, tb_log_name=model_name)

# Save the agent
model.save(model_dir + model_name)

print("training done")

rewards = np.zeros(n_eval_episodes)
current_reward, episode = 0, 0
env = swap_enviorment(depth_of_code, rows, cols, max_swaps_per_time_step)
while episode < n_eval_episodes:
    action = env.action_space.sample()
    _, reward, done, _ = env.step(action)
    current_reward += reward
    if done:
        rewards[episode] = current_reward
        current_reward = 0
        episode += 1
        env.reset()

print(f"Mean reward random: {np.mean(rewards)}")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)

print(f"Mean reward model: {mean_reward}")

write_text=open(f"logdir/{model_name}_{len(os.listdir(logdir))}/{model_name}"+".txt","w+")
write_text.write(
    f"Model: {model_name} \r\n"+
    "\r\n"+
    
    "Environment Variables \r\n"+
    f"depth_of_code = {depth_of_code}\r\n"+
    f"rows = {rows}\r\n"+
    f"cols = {cols}\r\n"+
    f"max_swaps_per_time_step = {max_swaps_per_time_step}\r\n"+
    "\r\n"+
    
    "Model Variables \r\n"+
    f"learning_starts = {learning_starts}\r\n"+
    f"verbose = {verbose}\r\n"+
    f"exploration_fraction = {exploration_fraction}\r\n"+
    f"exploration_initial_eps = {exploration_initial_eps}\r\n"+
    f"exploration_final_eps = {exploration_final_eps}\r\n"+
    f"batch_size = {batch_size}\r\n"+
    f"learning_rate = {learning_rate}\r\n"+
    f"target_update_interval = {target_update_interval}\r\n"+
    f"tau = {tau}\r\n"+
    f"gamma = {gamma}\r\n"+
    f"train_freq = {train_freq}\r\n"+
    "\r\n"+
    
    "Training Variables \r\n"+
    f"total_timesteps = {total_timesteps}\r\n"+
    f"log_interval = {log_interval}\r\n"+
    "\r\n"+
    
    "Evaluation \r\n"+
    f"n_eval_episodes = {n_eval_episodes}\r\n"+
    "\r\n"+
    
    "Results \r\n"+
    f"np.mean(rewards) = {np.mean(rewards)}\r\n"+
    f"mean_reward = {mean_reward}\r\n")

write_text.close()
