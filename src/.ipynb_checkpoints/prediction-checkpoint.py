import argparse

from models import *
from experiment import *

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn.init as init
from train import *

def create_sets(val_ds):
    X_input, X_target = val_ds[:-1], val_ds[1:]
    return X_input, X_target  

def make_predictions(X_input, X_target, prediction_steps=20):
    snapshots_pred, snapshots_truth = [], []
    error = []
    for i in range(30):
                error_temp = []
                init = X_input[i].float().to(device)
                if i == 0: init0 = init
                
                z = model.encoder(init) # embedd data in latent space

                for j in range(prediction_steps_steps):
                    if isinstance(z, tuple): z = model.dynamics(*z) # evolve system in time
                    else: z = model.dynamics(z)
                    if isinstance(z, tuple):
                        x_pred = model.decoder(z[0])
                    else:
                        x_pred = model.decoder(z) # map back to high-dimensional space
                    target_temp = X_target[i+j].data.cpu().numpy().reshape(m,n)
                    error_temp.append(np.linalg.norm(x_pred.data.cpu().numpy().reshape(m,n) - target_temp) / np.linalg.norm(target_temp))
                    
                    if i == 0:
                        snapshots_pred.append(x_pred.data.cpu().numpy().reshape(m,n))
                        snapshots_truth.append(target_temp)
    
                error.append(np.asarray(error_temp))


if __name__ == "__main__":
    # initialise
    datasetName = "pendulum"
    batchSize=128
    eigenInit = "gaussianElement"
    eigenLoss = "none"
    std=1.0
    epochs=10



    train_ds, val_ds, _, train_loader, val_loader, input_size, alpha, beta, lr = create_dataset(datasetName, batchSize)
    init_scheme = InitScheme(eigenInit, std, beta)
    model = create_model(alpha, beta, init_scheme, input_size)
    # model = koopmanAE(init_scheme, beta, alpha, input_size)
    loss_dict = train(model, 0, train_loader, val_loader, len(train_ds), len(val_ds), lr, eigenLoss, epochs)
    print(loss_dict)