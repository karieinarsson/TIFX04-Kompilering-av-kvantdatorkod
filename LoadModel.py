from MultiSwapEnviorment import swap_enviorment

from gym.envs.registration import register

import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from dqn.dqn import DQN
from dqn.evaluation import evaluate_policy

depth_of_code = 5
rows = 3
cols = 3
max_swaps_per_time_step = -1

n_eval_episodes = 1000

modelDir = "models/"

modelName = "DQNModel(StateToValue,10mil_step).zip"
#"DQNModel(StateToValue).zip"

register(
    id="MultiSwapEnviorment-v0",
    entry_point="MultiSwapEnviorment:swap_enviorment",
    max_episode_steps=200,
)

venv = make_vec_env("MultiSwapEnviorment-v0", n_envs=1, env_kwargs = {"depth_of_code": depth_of_code, "rows": rows, "cols": cols, "max_swaps_per_time_step": max_swaps_per_time_step})
# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load(modelDir + modelName, env=venv)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

#print(mean_reward)

env = venv.envs[0]

obs = venv.reset()
render_list = []
done = False
render_list.append(obs[0][0][0].tolist())

while not done:
    action, _states = model.predict(obs, venv,  deterministic=True)
    if action != 0:
        render_list.append(action[0])
    #only add first obs since it removes the first one to step 
    obs,_,done,_ = venv.step(action)
    render_list.append(obs[0][0][0].tolist())

env.render(mode = "human",render_list = render_list)

#rewards = np.zeros(n_eval_episodes)
#current_reward, episode = 0, 0
#env = swap_enviorment(depth_of_code, rows, cols, max_swaps_per_time_step)
#while episode < n_eval_episodes:
#    action = env.action_space.sample()
#    _, reward, done, _ = env.step(action)
#    current_reward += reward
#    if done:
#        rewards[episode] = current_reward
#        current_reward = 0
#        episode += 1
#        env.reset()
#
#print(f"Mean reward random: {np.mean(rewards)} +/- {np.std(rewards)}")
#
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
#print(f"Mean reward model: {mean_reward} +/- {std_reward}")
