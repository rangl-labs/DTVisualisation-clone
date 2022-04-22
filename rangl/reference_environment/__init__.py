from gym.envs.registration import register

register(
    id="rangl-NBMdata-v0",
    entry_point="reference_environment.env:GymEnv",
)
