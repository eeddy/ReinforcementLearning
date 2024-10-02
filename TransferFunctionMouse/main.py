from tf_gym import TFGym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from custom_policy import * 

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)
env = TFGym('human')

model = PPO(CustomActorCriticPolicy, env, verbose=1)
model.learn(total_timesteps=100000, log_interval=5000, callback=checkpoint_callback)

# model = PPO.load('logs/rl_model_15000_steps.zip')

# env = TFGym('human')
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, reward, terminated, _, _ = env.step(action)
#     if terminated:
#         obs, info = env.reset()