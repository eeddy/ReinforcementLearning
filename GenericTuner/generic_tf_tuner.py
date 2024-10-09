from abc import ABC, abstractmethod
from tf_gym import *
from tf_gym_emg import * 
import socket
import torch as th
from pretraining import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from multiprocessing import Process

class InputDevice(ABC):
    """
    This is the base class for any input device. 

    low:        the lowest reading from the device. 
    high:       the highest possible reading from the device.
    in_shape:   the input shape to the network (e.g., a pointing device with dx and dy would be 2).
    out_shape:  the output shape from the network (typically this should equal in_shape.)
    """
    def __init__(self, low, high, in_shape=2, out_shape=2, port=12345, name='device'):
        self.low = low 
        self.high = high 
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.port = port 
        self.name = name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @abstractmethod
    def run(self):
        pass

class TFEnvironment:
    """
    This is the base class for creating a training environment for a given input device.

    device:     an object that inherits from input device.
    actions:    a dictionary with the following keys: 'high', 'low'. Defines the lower and upper bounds of the action space. 
    pretrain:   if not none, this should be a dictionary with the following keys 'function', 'epochs', 'samples'. 
    timesteps:  the number of timesteps to run the RL for.
    save_freq:  how many timesteps before saving the model.
    save_path:  where to save the model.
    """
    
    def __init__(self, device, actions, pretrain=None, timesteps=50000, save_freq=1000, save_path=None, emg=False):
        self.device = device 
        self.actions = actions 
        self.pretrain = pretrain 
        self.timesteps = timesteps
        self.emg = emg

        if save_path == None:
            save_path = './logs/' + self.device.name

        # Callback 
        self.checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=self.device.name,
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
    
    def run(self, evaluate=None):
        # Start streaming from device 
        p = Process(target=self.device.run)
        p.start()

        obs_space = spaces.Box(low=self.device.low, high=self.device.high, shape=(self.device.in_shape,), dtype=np.float32)
        act_space = spaces.Box(low=self.actions['low'], high=self.actions['high'], shape=(self.device.out_shape,), dtype=np.float32)

        if evaluate is not None:
            print("Starting in evaluation mode!")
            model = PPO.load(evaluate)
            # Start training PPO 
            if self.emg:
                env = TFGymEMG(obs_space, act_space)
            else:
                env = TFGym(obs_space, act_space)

            obs, _ = env.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, _, _ = env.step(action)
                if terminated:
                    obs, _ = env.reset()
        else:
            # Start training PPO 
            if self.emg:
                env = TFGymEMG(obs_space, act_space)
            else:
                env = TFGym(obs_space, act_space)

            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[64,64], vf=[64,64]))

            model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)

            # Check if we need to pretrain 
            if self.pretrain is not None:
                ds = generate_dataset(self.device.low, self.device.high, self.pretrain['func'], self.pretrain['samples'])
                fit(model.policy, ds, num_epochs=self.pretrain['epochs'])

            model.learn(total_timesteps=self.timesteps, callback=self.checkpoint_callback, log_interval=10_000)





    

