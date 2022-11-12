import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import xarray
from pathlib import Path
import dask
from tqdm import tqdm
import scipy.io

dask.config.set(scheduler='synchronous')

class PendulumToPendulum(Dataset):
    def __init__(self, prediction_length, dissipation_level, partition_name='train'):
        self.pendulum_array = np.load(f"pendulum_dissipative_{partition_name}_long.npy")
        self.dissipation_level = dissipation_level
        self.prediction_length = prediction_length
    
    def __len__(self):
        return len(self.pendulum_array[self.dissipation_level])
    
    def __getitem__(self, idx):
        i = 0
        for pend_run in self.pendulum_array[self.dissipation_level]:
            j = self.prediction_length
            for time_step in pend_run[self.prediction_length:-self.prediction_length]:
                if i == idx:
                    return torch.from_numpy(pend_run[j-self.prediction_length:j+self.prediction_length]), torch.from_numpy(np.flip(pend_run[j-self.prediction_length:j+self.prediction_length], 0).copy())
                j += 1
                i += 1

def generate_pendulum_ds(dissipation_level, prediction_horizon=96):
    train_ds = PendulumToPendulum(prediction_horizon, dissipation_level, 'train')
    val_ds = PendulumToPendulum(prediction_horizon, dissipation_level, 'valid')
    test_ds = PendulumToPendulum(prediction_horizon, dissipation_level, 'test')

    return train_ds, val_ds, test_ds, prediction_horizon
