from MultiSwapEnviorment import SwapEnviorment

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

depthOfCode = 10
rows = 2
cols = 2
trainingSteps = int(1e5)
verbose = 1


#create the enviorment
env = SwapEnviorment(depthOfCode, rows, cols)

# Intantiate the agent
model = DQN('MlpPolicy', env, verbose=verbose)

# Train the agent
model.learn(total_timesteps = trainingSteps)

# Save the agent
modelDir = "models/"
modelName = "DQNModel("+ str(env.depthOfCode) + ", " + str(env.rows) + ", " + str(env.cols) + ")"

model.save(modelDir + modelName)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward)
