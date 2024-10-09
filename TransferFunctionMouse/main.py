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

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[32, 16], vf=[32, 16]))

model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
ds, data = generate_dataset(negative=False)
fit(model.policy, ds, num_epochs=10)
model.learn(total_timesteps=50_000, callback=checkpoint_callback, log_interval=10_000)

# model = PPO.load('logs/rl_model_20000_steps.zip')

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, _, _ = env.step(action)
#     if terminated:
#         obs, info = env.reset()