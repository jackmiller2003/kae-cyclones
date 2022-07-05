import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings("ignore")

def eigen_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())
    w = np.random.uniform(-1,1, w.shape[0])
    return torch.from_numpy(reconstruct_operator(w,v).real).float()

def reconstruct_operator(w, v):
    """
    Recreate a matrix from its eigenvalues and eigenvectors.
    """
    R = np.linalg.inv(v)
    # create diagonal matrix from eigenvalues
    L = np.diag(w)
    # reconstruct the original matrix
    B = v.dot(L).dot(R)
    return B

class Autoencoder(nn.Module):
    def __init__(self, eigen_init=False, b=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.linear = nn.Sequential(
            nn.Linear(64, b),
            nn.Linear(b, b),
            nn.Linear(b, 64)
        )
        if eigen_init:
            self.linear[1].weight.data = eigen_init_(b, std=1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 64)
        x = self.linear(x)
        x = torch.reshape(x, (x.size(0), 64, 1, 1))
        x = self.decoder(x)
        return x