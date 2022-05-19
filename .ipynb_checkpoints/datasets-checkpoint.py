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

train_json_path = '/g/data/x77/ob2720/partition/train.json'
valid_json_path = '/g/data/x77/ob2720/partition/valid.json'
test_json_path = '/g/data/x77/ob2720/partition/test.json'

dask.config.set(scheduler='synchronous')

class CycloneToCycloneDataset(Dataset):
    def __init__(self, cyclone_dir, json_path):
        with open(json_path, 'r') as jp:
            self.tracks_dict = json.load(jp)
        
        self.cyclone_dir = cyclone_dir
        
    def __len__(self):
        length = 0
            
        return len(self.tracks_dict.keys())
    
    def __getitem__(self, idx):
        i = 0

        for cyclone, data in self.tracks_dict.items():            
            if i == idx:
                cyclone_ds = xarray.open_dataset(self.cyclone_dir+cyclone+".nc", engine='netcdf4', cache=False, chunks='auto')
                time_length = len(cyclone_ds.time)
                example = cyclone_ds.to_array().to_numpy()
                example = example.transpose((1,0,2,3,4))
                example = np.reshape(example, (time_length,20,160,160))

                return torch.from_numpy(example)
                
            i += 1


def generate_example_dataset():
    return CycloneToCycloneDataset('/g/data/x77/ob2720/partition/train/', train_json_path)

