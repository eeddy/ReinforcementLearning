from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from itertools import combinations
from tf_gym import *

fig, axs = plt.subplots(2)
steps = [5000, 25000, 50000, 75000]

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

    arrx = []
    arry = []
    for i in range(0, 140):
        arrx.append(model.predict(np.array([i,0]), deterministic=True)[0])
        arry.append(model.predict(np.array([0,i]), deterministic=True)[0])
    
    arrx = np.array(arrx)
    arry = np.array(arry)

    axs[0].plot(arrx[:,0],  label=s, linewidth=3)
    axs[1].plot(arry[:,1],  label=s, linewidth=3)

    if s_i == len(steps)-1:
        np.save('transfer_func_x.npy', arrx[:,0])
        np.save('transfer_func_y.npy', arry[:,1])

axs[0].set_xlabel('Counts')
axs[1].set_xlabel('Counts')
axs[0].set_ylabel('Gain')
axs[1].set_ylabel('Gain')
axs[0].set_title('DX')
axs[1].set_title('DY')
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()