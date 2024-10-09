from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
import random
import numpy as np
from torch import nn
from sklearn.metrics import mean_squared_error
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class DL_input_data(Dataset):
    def __init__(self, x, y):
        self.x = th.tensor(x, dtype=th.float32)
        self.y = th.tensor(y, dtype=th.float32)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def __len__(self):
        return self.x.shape[0]

def make_data_loader(x, y, batch_size=1000):
    obj = DL_input_data(x, y)
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True)
    return dl

def generate_dataset(min, max, func, samples=100000):
    y = []
    x = []
    for _ in range(0, samples):
        x_val = random.randrange(min, max) # TODO: Fix this so that it works with any input size
        y_val = random.randrange(min, max)
        x.append(np.array([x_val]))
        y.append(func(x[-1]))
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()
    return make_data_loader(x, y)

def fit(network, tr_dl, learning_rate=1e-3, num_epochs=50, verbose=True):
        device = 'cuda' if th.cuda.is_available() else 'cpu'

        network.to(device)
        # get the optimizer and loss function ready
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        # Logger:
        log = {"training_loss":[],
               "training_mse": []} 
        # now start the training
        for epoch in range(num_epochs):
            #training set
            network.train()
            for data, labels in tr_dl:
                optimizer.zero_grad()
                data = data.to(device)
                labels = labels.to(device)
                output = network.forward(data, deterministic=True)[0]
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                mse = mean_squared_error(output.to('cpu').detach().numpy(), labels.to('cpu').detach().numpy())

                log["training_loss"] += [(epoch, loss.item())]
                log["training_mse"] += [(epoch, mse)]
            if verbose:
                epoch_trloss = np.mean([i[1] for i in log['training_loss'] if i[0]==epoch])
                epoch_tracc  = np.mean([i[1] for i in log['training_mse'] if i[0]==epoch])
                print(f"{epoch}: trloss:{epoch_trloss:.2f}  trmse:{epoch_tracc:.2f}")
        network.eval()
        return network