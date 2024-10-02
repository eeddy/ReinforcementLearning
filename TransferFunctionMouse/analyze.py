from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from tf_gym import *

steps = [1000, 5000, 30000]
labels = ['~5s', '~25s', '~5 Minutes']

for s_i, s in enumerate(steps):
    file = 'logs/rl_model_' + str(s) + '_steps.zip'

    if s == 0: 
        env = TFGym('human')
        model = PPO('MlpPolicy', env, verbose=1)
        # I want to update the policy 
        ds, data = generate_dataset()
        fit(model.policy, ds, num_epochs=5) 
    else:
        model = PPO.load(file)

    arr = []
    for i in range(0, 120):
        arr.append(model.predict(np.array([i,1]), deterministic=True)[0])

    arr = np.array(arr)
    plt.plot(arr[:,0], label=labels[s_i], linewidth=3)

plt.legend()
plt.ylabel('% Max Speed')
plt.xlabel('# Counts')
plt.show()
print("HERE")