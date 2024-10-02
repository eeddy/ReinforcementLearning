from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 

steps = [1000, 15000]

for s in steps:
    file = 'logs/rl_model_' + str(s) + '_steps.zip'

    model = PPO.load(file)

    arr = []
    for i in range(0, 2000):
        arr.append(model.predict(np.array([i,0]))[0])

    arr = np.array(arr)
    plt.plot(arr[:,0], label='x')

plt.show()
print("HERE")