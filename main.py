from fitts_gym import FittsEnv

# env = FittsEnv('human')
# obs = env.reset()
# while True: 
#     random_action = env.action_space.sample()
#     obs, reward, done, _, _ = env.step(random_action)
#     if done:
#         obs, info = env.reset()

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
        