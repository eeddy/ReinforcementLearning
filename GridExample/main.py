from grid_world import GridWorldEnv

# env = GridWorldEnv('human')

# for episode in range(5):
# 	print(episode)
# 	done = False
# 	obs = env.reset()
# 	while True:#not done:
# 		random_action = env.action_space.sample()
# 		obs, reward, done, _, _ = env.step(random_action)
# 		print(reward)

from stable_baselines3 import PPO

env = GridWorldEnv('human')

model = PPO("MultiInputPolicy", env, verbose=1, seed=1)
model.learn(total_timesteps=100000, log_interval=4)

env = GridWorldEnv('human')
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, _, _ = env.step(action)
    if terminated:
        obs, info = env.reset()