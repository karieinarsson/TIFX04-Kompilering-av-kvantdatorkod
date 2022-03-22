from MultiSwapEnviorment import swap_enviorment

import numpy as np
import os
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from dqn.dqn import DQN

#env variables
depth_of_code = 2
rows = 3
cols = 3
max_swaps_per_time_step = -1

#model variables
learning_starts = int(1e5)
verbose = 1
exploration_fraction = 0.5
exploration_initial_eps = 1
exploration_final_eps = 0.1
batch_size = 128
learning_rate = 0.001
tau = 0.5
gamma = 0.99
train_freq = 10

#training variables
total_timesteps = int(5e5)
log_interval = 10


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              #if self.verbose > 0:
                #print(f"Num timesteps: {self.num_timesteps}")
                #print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  #if self.verbose > 0:
                    #print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = swap_enviorment(depth_of_code, rows, cols, max_swaps_per_time_step)

env = Monitor(env, log_dir)

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

#create the enviorment

# Intantiate the agent
model = DQN('CnnPolicy', 
            env, 
            verbose = verbose,
            train_freq = train_freq,
            gamma = gamma,
            tau = tau,
            learning_starts = learning_starts, 
            exploration_fraction = exploration_fraction, 
            exploration_final_eps = exploration_final_eps, 
            exploration_initial_eps = exploration_initial_eps,
            batch_size = batch_size,
            optimize_memory_usage = True,
            learning_rate = learning_rate
        )

# Train the agent
model.learn(total_timesteps = total_timesteps, log_interval = log_interval, callback = callback)

# Save the agent
model_dir = "models/"
model_name = "DQNModel("+ str(env.depth_of_code) + "," + str(env.rows) + "," + str(env.cols) + ',' + str(env.max_swaps_per_time_step) + ")"

model.save(model_dir + model_name)


# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "DQN")
plt.show()

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print(mean_reward)

