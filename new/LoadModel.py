from MultiSwapEnviorment import SwapEnviorment

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


modelDir = "models/"

modelName = "DQNModel(1,3,3)"

env = SwapEnviorment(1, 3, 3, maxSwapsPerTimeStep = 1)

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load(modelDir + modelName, env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

print(mean_reward)

