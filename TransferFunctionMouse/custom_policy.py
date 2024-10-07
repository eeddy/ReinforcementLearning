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

def generate_dataset():
    y = []
    x = []
    for _ in range(0, 1000000):
        x_val = random.randrange(0, 140)
        if random.random() > 0.5:
            x_val = -x_val
        y_val = random.randrange(0, 140)
        if random.random() > 0.5:
            y_val = -y_val
        x.append(np.array([x_val, y_val]))
        y.append(np.abs(x[-1])/120)
    return make_data_loader(x, y), [x,y]

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
                mse = mean_squared_error(output.detach().numpy(), labels.detach().numpy())

                log["training_loss"] += [(epoch, loss.item())]
                log["training_mse"] += [(epoch, mse)]
            if verbose:
                epoch_trloss = np.mean([i[1] for i in log['training_loss'] if i[0]==epoch])
                epoch_tracc  = np.mean([i[1] for i in log['training_mse'] if i[0]==epoch])
                print(f"{epoch}: trloss:{epoch_trloss:.2f}  trmse:{epoch_tracc:.2f}")
        network.eval()
        return network

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU(),
        )
        # Train my polciy network to produce y=x
        # print('Training Policy Network')
        # dl = generate_dataset()
        # fit(self.policy_net, dl, learning_rate=1e-3)
        # print('Pre-trained Policy')


        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)