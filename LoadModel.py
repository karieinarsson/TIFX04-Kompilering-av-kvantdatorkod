from MultiSwapEnviorment_2 import swap_enviorment

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


modelDir = "tmp/"

modelName = "best_model.zip"

env = swap_enviorment(1, 3, 3, 2)

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

