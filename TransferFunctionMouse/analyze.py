from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from tf_gym import *

fig, axs = plt.subplots(2,2)
steps = [1000, 5000, 10000, 25000]

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
    for i in range(0, 120):
        arrx.append(model.predict(np.array([i,1]), deterministic=True)[0])
        arry.append(model.predict(np.array([1,i]), deterministic=True)[0])

    arrx = np.array(arrx)
    arry = np.array(arry)
    axs[0,0].plot(arrx[:,0], label=s, linewidth=3)
    axs[1,0].plot(arry[:,1], label=s, linewidth=3)

    arrx = []
    arry = []
    for i in range(0, 120):
        arrx.append(model.predict(np.array([-i,1]), deterministic=True)[0])
        arry.append(model.predict(np.array([1,-i]), deterministic=True)[0])

    arrx = np.array(arrx)
    arry = np.array(arry)
    axs[0,1].plot(arrx[:,0], label=s, linewidth=3)
    axs[1,1].plot(arry[:,1], label=s, linewidth=3)

axs[0,0].set_title('+DX')
axs[0,1].set_title('-DX')
axs[1,0].set_title('+DY')
axs[1,1].set_title('-DY')

for i in [[0,0], [0,1], [1,0], [1,1]]:
    axs[*i].set_ylabel('% Max Speed')
    axs[*i].set_xlabel('# Counts')
    axs[*i].legend()
plt.tight_layout()
plt.show()
print("HERE")