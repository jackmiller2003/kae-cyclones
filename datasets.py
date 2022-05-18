import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import xarray
from pathlib import Path

train_json_path = '/g/data/x77/ob2720/partition/train.json'
valid_json_path = '/g/data/x77/ob2720/partition/valid.json'
test_json_path = '/g/data/x77/ob2720/partition/test.json'

class CycloneToCycloneDataset(Dataset):
    def __init__(self, json_path, prediction_length):
        with open(json_path, 'r') as jp:
            self.tracks_dict = json.load(jp)
        
        self.prediction_length = prediction_length
        
    def __len__(self):
        length = 0
        for cyclone, data in self.tracks_dict.items():
            if len(data['coordinates']) > self.prediction_length:
                length += len(data['coordinates'][:-self.prediction_length])
    
    def __getitem__(self, idx):
        i = 0

        for cyclone, data in self.tracks_dict.items():
            j = 0

            if len(data['coordinates']) > self.prediction_length:
                for coordinate in data['coordinates'][:-self.prediction_length]:
                    if i == idx:
                        cyclone_ds = xarray.open_dataset(self.cyclone_dir+cyclone+".nc", 
                            engine='netcdf4', decode_cf=False, cache=True)
                        
                        cyclone_ds[dict(time=[j, j+self.prediction_length])]

                        print(cyclone_ds)

if __name__ == "__main__":
    example_dataset = CycloneToCycloneDataset(train_json_path, 1)
    e = example_dataset[0]

