from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from itertools import combinations
from tf_gym import *

def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data = np.load('rewards.npy')
plt.plot(moving_average(data), linewidth=4)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()

def convert_to_speed(x):
    return x / 3200 * 0.0254 * 125 

fig, axs = plt.subplots(2)
steps = [25000]

for s_i, s in enumerate(steps):
    file = 'logs/mouse_small/mouse_' + str(s) + '_steps.zip'
    model = PPO.load(file)

    arrx = []
    arry = []
    x = []
    for i in range(0, 40):
        arrx.append(convert_to_speed(model.predict(np.array([i,0]), deterministic=True)[0] * i))
        arry.append(convert_to_speed(model.predict(np.array([0,i]), deterministic=True)[0] * i))
        x.append(convert_to_speed(i))
    
    arrx = np.array(arrx)
    arry = np.array(arry)

    axs[0].plot(x, arrx[:,0], linewidth=3)
    axs[1].plot(x, arry[:,1], linewidth=3)

    if s_i == len(steps)-1:
        np.save('transfer_func_x.npy', arrx[:,0])
        np.save('transfer_func_y.npy', arry[:,1])

axs[0].set_xlabel('In Speed (m/s)')
axs[1].set_xlabel('In Speed (m/s)')
axs[0].set_ylabel('Out Speed (m/s)')
axs[1].set_ylabel('Out Speed (m/s)')
axs[0].set_title('DX')
axs[1].set_title('DY')
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()