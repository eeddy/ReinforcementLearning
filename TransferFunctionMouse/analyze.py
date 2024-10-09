from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from itertools import combinations
from tf_gym import *

def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data = np.load('rewards.npy')
plt.plot(moving_average(data), linewidth=4)
plt.xlabel('Episodes')
plt.ylabel('Throughput (bits/s)')
plt.show()

fig, axs = plt.subplots(2)
steps = [1000, 2000, 5000]

for s_i, s in enumerate(steps):
    file = 'logs/mouse/mouse_' + str(s) + '_steps.zip'
    model = PPO.load(file)

    arrx = []
    arry = []
    x = []
    for i in range(0, 140):
        arrx.append(model.predict(np.array([i,0]), deterministic=True)[0] * i)
        arry.append(model.predict(np.array([0,i]), deterministic=True)[0] * i)
        x.append(i)
    
    arrx = np.array(arrx)
    arry = np.array(arry)

    axs[0].plot(x, arrx[:,0],  label=s, linewidth=3)
    axs[1].plot(x, arry[:,1],  label=s, linewidth=3)

    if s_i == len(steps)-1:
        np.save('transfer_func_x.npy', arrx[:,0])
        np.save('transfer_func_y.npy', arry[:,1])

axs[0].set_xlabel('# of Pixels')
axs[1].set_xlabel('# of Pixels')
axs[0].set_ylabel('Gain')
axs[1].set_ylabel('Gain')
axs[0].set_title('DX')
axs[1].set_title('DY')
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()