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

train_json_path = '/g/data/x77/ob2720/partition/train.json'
valid_json_path = '/g/data/x77/ob2720/partition/valid.json'
test_json_path = '/g/data/x77/ob2720/partition/test.json'

dask.config.set(scheduler='synchronous')

class CycloneToCycloneDataset(Dataset):
    def __init__(self, cyclone_dir, json_path, prediction_length, atmospheric_values, pressure_levels,
                load_np=True, save_np=False, partition_name='train', crop=True):
        with open(json_path, 'r') as jp:
            self.tracks_dict = json.load(jp)

        self.cyclone_dir = cyclone_dir
        self.prediction_length = prediction_length
        self.atmospheric_values = atmospheric_values
        self.pressure_levels = pressure_levels
        self.load_np = load_np
        self.save_np = save_np
        self.partition_name = partition_name
        self.crop = crop

    def __len__(self):
        length = 0
        for cyclone, data in self.tracks_dict.items():
            if len(data['coordinates']) > 2*self.prediction_length:
                length += len(data['coordinates'][self.prediction_length:-self.prediction_length])
        
        return length

    def __getitem__(self, idx):
        i = 0

        for cyclone, data in self.tracks_dict.items():
            j = self.prediction_length

            if len(data['coordinates']) > 2*self.prediction_length:
                for coordinate in data['coordinates'][self.prediction_length:-self.prediction_length]:
                    if i == idx:
                        if self.save_np:
                            cyclone_ds = xarray.open_dataset(self.cyclone_dir+cyclone+".nc", 
                            engine='netcdf4', decode_cf=False, cache=True)

                            cyclone_ds = cyclone_ds[dict(time=list(range(-self.prediction_length+j, self.prediction_length+j)),
                                                    level=self.pressure_levels)][self.atmospheric_values]

                            cyclone_array = cyclone_ds.to_array().to_numpy()
                            
                            size = 160

                            if self.crop:
                                cyclone_array = cyclone_array[:,:,:,40:120,40:120]
                                cyclone_array = cyclone_array[:,:,:,::4,::4]
                                size = 20

                            cyclone_array = cyclone_array.transpose((1,0,2,3,4))
                            cyclone_array = cyclone_array.reshape(self.prediction_length*2, -1, size, size)                          

                            return (cyclone_array, cyclone, j)

                        if self.load_np:
                            if self.crop:
                                cyclone_array = np.load(f'/g/data/x77/jm0124/np_cyclones_crop/{self.prediction_length}/{self.partition_name}/{cyclone}-{j}.npy') * 0.004376953827249367
                                reversed_array = np.flip(cyclone_array, 0).copy() * 0.004376953827249367
                            else:
                                cyclone_array = np.load(f'/g/data/x77/jm0124/np_cyclones/{self.prediction_length}/{self.partition_name}/{cyclone}-{j}.npy') * 0.004376953827249367
                                reversed_array = np.flip(cyclone_array, 0).copy() * 0.004376953827249367

                            return torch.from_numpy(cyclone_array), torch.from_numpy(reversed_array)

                    j += 1
                    i += 1

def generate_example_dataset():
    train_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/train/', train_json_path, 2,
                                        ['u'], [0], load_np=True, save_np=False, partition_name='train')
    val_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/valid/', valid_json_path, 2,
                                        ['u'], [0], load_np=True, save_np=False, partition_name='valid')
    test_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/test/', test_json_path, 2,
                                        ['u'], [0], load_np=True, save_np=False, partition_name='test')

    return train_ds, val_ds, test_ds

def generate_numpy_dataset(prediction_length, atmospheric_values, pressure_levels):
    train_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/train/', train_json_path, prediction_length,
                                        atmospheric_values, pressure_levels, load_np=False, save_np=True, partition_name='train')
    val_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/valid/', valid_json_path, prediction_length,
                                        atmospheric_values, pressure_levels, load_np=False, save_np=True, partition_name='valid')
    test_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/test/', test_json_path, prediction_length,
                                        atmospheric_values, pressure_levels, load_np=False, save_np=True, partition_name='test')

    # print("Train ds")
    # for i,(cyclone_array, cyclone, j) in tqdm(enumerate(train_ds)):
    #     np.save(f'/g/data/x77/jm0124/np_cyclones_crop/{prediction_length}/train/{cyclone}-{j}.npy', cyclone_array)
    
    print("Val ds")
    for i,(cyclone_array, cyclone, j) in tqdm(enumerate(val_ds)):
        np.save(f'/g/data/x77/jm0124/np_cyclones_crop/{prediction_length}/valid/{cyclone}-{j}.npy', cyclone_array)

    print(len(val_ds))

    print("Test ds")
    for i,(cyclone_array, cyclone, j) in tqdm(enumerate(test_ds)):
        np.save(f'/g/data/x77/jm0124/np_cyclones_crop/{prediction_length}/test/{cyclone}-{j}.npy', cyclone_array)

if __name__ == '__main__':
    generate_numpy_dataset(2, ['u'], [2])
