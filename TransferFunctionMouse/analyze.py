from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from itertools import combinations
from tf_gym import *

def convert_to_speed(val, dpi=3200, refresh=125):
    return val/dpi * 0.0254 * refresh # 0.0254 is inches to meter

fig, axs = plt.subplots(2)
steps = [5000, 10000, 20000]

for s_i, s in enumerate(steps):
    file = 'logs/rl_model_' + str(s) + '_steps.zip'
    model = PPO.load(file)

    arrx = []
    arry = []
    x = []
    for i in range(0, 140):
        speed = convert_to_speed(i)
        arrx.append(model.predict(np.array([speed,0]), deterministic=True)[0])
        arry.append(model.predict(np.array([0,speed]), deterministic=True)[0])
        x.append(speed)
    
    arrx = np.array(arrx)
    arry = np.array(arry)

    axs[0].plot(x, arrx[:,0],  label=s, linewidth=3)
    axs[1].plot(x, arry[:,1],  label=s, linewidth=3)

    # if s_i == len(steps)-1:
    #     np.save('transfer_func_x.npy', arrx[:,0])
    #     np.save('transfer_func_y.npy', arry[:,1])

axs[0].set_xlabel('Speed (m/s)')
axs[1].set_xlabel('Speed (m/s)')
axs[0].set_ylabel('Gain')
axs[1].set_ylabel('Gain')
axs[0].set_title('DX')
axs[1].set_title('DY')
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()