from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from custom_policy import * 
from itertools import combinations
from tf_gym import *

# fig, axs = plt.subplots(2,2)
steps = [1000, 10000, 25000, 35000, 50000, 57000]

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

    tf = np.mean([arrx[:,0], arry[:,1]], axis=0)
    plt.plot(tf,  label=s, linewidth=3)

    if s_i == len(steps)-1:
        np.save('transfer_func.npy', tf)

plt.legend()
plt.show()