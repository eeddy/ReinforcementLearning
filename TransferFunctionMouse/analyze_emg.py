from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from itertools import combinations
from tf_gym import *

steps = [1000, 5000, 8000]

def convert_to_speed(x):
    return x * 50 * 40  / 400 * 0.0254 # Assuming PPI is approximately 400

for s_i, s in enumerate(steps):
    file = 'logs/emg/emg_' + str(s) + '_steps.zip'
    model = PPO.load(file)

    y = []
    x = []
    for i in range(0, 128*5):
        x.append(i/5)
        y.append(convert_to_speed(model.predict(np.array([i/5]), deterministic=True)[0]))

    plt.plot(x, y, label = s, linewidth=4)

plt.xlabel('MAV')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()