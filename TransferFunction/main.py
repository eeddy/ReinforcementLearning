from tf_gym import TFGym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# # Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(
#   save_freq=1000,
#   save_path="./logs/",
#   name_prefix="rl_model",
#   save_replay_buffer=True,
#   save_vecnormalize=True,
# )
# env = TFGym('human')

# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=100000, log_interval=4, callback=checkpoint_callback)

# env = TFGym('human')
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, _, _ = env.step(action)
#     if terminated:
#         obs, info = env.reset()


model = PPO.load('logs/rl_model_54000_steps.zip')

env = TFGym('human')
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, _, _ = env.step(action)
    if terminated:
        obs, info = env.reset()