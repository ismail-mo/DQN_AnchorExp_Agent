from gymnasium.envs.registration import register

register(
    id="AnchorExpEnv-v0",
    entry_point="env.AnchorExp_env:AnchorExpEnv",
    max_episode_steps=200,
)