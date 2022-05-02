from gym.envs.registration import register
#from MultiSwap import swap_environment
register(
    id="MultiSwapEnviorment-v0",
    entry_point="MultiSwap:swap_enviorment",
    max_episode_steps=200,
)

