from MultiSwapEnviorment import swap_enviorment

from gym.envs.registration import register

from dqn.env_util import make_vec_env
from dqn.dqn import DQN
from dqn.evaluation import evaluate_policy

depth_of_code = 10
rows = 2
cols = 2
max_swaps_per_time_step = -1

modelDir = "models/"

modelName = "DQNModel(StateToValue).zip"

register(
    id="MultiSwapEnviorment-v0",
    entry_point="MultiSwapEnviorment:swap_enviorment",
    max_episode_steps=200,
)

venv = make_vec_env("MultiSwapEnviorment-v0", env_kwargs = {"depth_of_code": depth_of_code, "rows": rows, "cols": cols, "max_swaps_per_time_step": max_swaps_per_time_step})

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load(modelDir + modelName, env=venv)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

print(mean_reward)

