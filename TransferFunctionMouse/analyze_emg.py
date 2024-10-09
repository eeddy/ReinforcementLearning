from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from itertools import combinations
from tf_gym import *

steps = [1000, 5000, 8000, 15000]

for s_i, s in enumerate(steps):
    file = 'logs/emg/emg_' + str(s) + '_steps.zip'
    model = PPO.load(file)

    arr = []
    for i in range(0, 128*5):
        arr.append(model.predict(np.array([i/5]), deterministic=True)[0])

    plt.plot(arr, label = s, linewidth=4)

plt.xlabel('MAV')
plt.ylabel('# of Pixels')
plt.legend()
plt.show()