from fitts_gym import FittsEnv
from stable_baselines3 import PPO

env = FittsEnv('human')

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)

env = FittsEnv('human')
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, _, _ = env.step(action)
    if terminated:
        obs, info = env.reset()
        