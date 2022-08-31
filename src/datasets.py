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
                            engine='netcdf4', decode_cf=True, cache=True) # parallel, cunk

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
                                cyclone_array = np.load(f'/g/data/x77/jm0124/np_cyclones_crop/{self.prediction_length}/{self.partition_name}/{cyclone}-{j}.npy')
                                reversed_array = np.flip(cyclone_array, 0).copy()
                            else:
                                cyclone_array = np.load(f'/g/data/x77/jm0124/np_cyclones/{self.prediction_length}/{self.partition_name}/{cyclone}-{j}.npy')
                                reversed_array = np.flip(cyclone_array, 0).copy()

                            return torch.from_numpy(cyclone_array), torch.from_numpy(reversed_array)

                    j += 1
                    i += 1

class CycloneDataset(Dataset):
    """
    Custom dataset for cyclones.
    """

    def __init__(self, cyclone_dir, tracks_path, save_np = False, load_np = True, crop = True, prediction_length = 0,
                atmospheric_values = ['u'], pressure_levels=[2], partition_name='train', synthetic=False, 
                synthetic_type='base_synthesis', sigma=0.1):
        self.cyclone_dir = cyclone_dir
        self.atmospheric_values = atmospheric_values
        self.pressure_levels = pressure_levels
        self.tracks_path = tracks_path
        self.load_np = load_np
        self.save_np = save_np
        self.crop = crop
        self.prediction_length = prediction_length
        self.partition_name = partition_name
        self.synthetic_type = synthetic_type
        self.sigma = sigma
        self.synthetic = synthetic

        with open(tracks_path, 'r') as jp:
            self.tracks_dict = json.load(jp)

    def __len__(self):
        if self.synthetic:
            bound = 2
        else:
            bound = 1
        length = 0
        for cyclone, data in self.tracks_dict.items():
            if len(data['coordinates']) > bound:
                length += len(data['coordinates'][:-bound])
        return length       
    
    def __getitem__(self, idx):
        i = 0

        for cyclone, data in self.tracks_dict.items():
            if self.synthetic:
                j = 1
                bound = 2
            else:
                j = 0
                bound = 1

            if len(data['coordinates']) > bound:
                for coordinate in data['coordinates'][:-bound]:
                    if i == idx:
                        if self.save_np:
                            cyclone_ds = xarray.open_dataset(self.cyclone_dir+cyclone+".nc", 
                            engine='netcdf4', decode_cf=True, cache=True)

                            cyclone_ds = cyclone_ds[dict(time=[j],
                                                    level=self.pressure_levels)][self.atmospheric_values]

                            cyclone_array = cyclone_ds.to_array().to_numpy()

                            if self.crop:
                                cyclone_array = cyclone_array[:,:,:,40:120,40:120]
                                cyclone_array = cyclone_array[:,:,:,::4,::4]
                                size = 20

                            cyclone_array = cyclone_array.transpose((1,0,2,3,4))
                            cyclone_array = cyclone_array.reshape(1, -1, size, size) 

                            return cyclone_array, cyclone, j

                        if self.load_np:
                            if self.synthetic:
                                try:
                                    cyclone_array = np.load(f'/g/data/x77/jm0124/synthetic_datasets/{self.synthetic_type}/{self.atmospheric_values[0]}/{self.pressure_levels[0]}/{self.sigma}/{self.partition_name}/{cyclone}-{j}.npy', \
                                                            allow_pickle=True)
                                except:
                                    print(f'{cyclone}-{j}')
                            else:
                                cyclone_array = np.load(f'/g/data/x77/jm0124/np_cyclones_crop/{self.prediction_length}/{self.atmospheric_values[0]}/{self.pressure_levels[0]}/{self.partition_name}/{cyclone}-{j}.npy')
                            
                            label = torch.from_numpy(np.array([
                                                        [float(data['coordinates'][j][0]), float(data['coordinates'][j+1][0])], 
                                                        [float(data['coordinates'][j][1]), float(data['coordinates'][j+1][1])]
                                                                ]))

                            return (torch.from_numpy(cyclone_array), label)

                    i += 1
                    j += 1

############################################################################
# OCEAN
############################################################################

class OceanToOcean(Dataset):
    def __init__(self, prediction_length, partition_name='train'):
        # self.ocean_array = np.load(f"/home/156/cn1951/kae-cyclones/input/sstday_{partition_name}.npy")
        self.ocean_array = np.load(f"/g/data/x77/jm0124/ocean/sstday_{partition_name}.npy")
        self.prediction_length = prediction_length
    
    def __len__(self):
        return self.ocean_array.shape[0]
    
    def __getitem__(self, idx):
        i = 0
        for ocean_run in self.ocean_array:
            j = self.prediction_length
            for time_step in ocean_run[self.prediction_length:-self.prediction_length]:
                if i == idx:
                    return torch.from_numpy(ocean_run[j-self.prediction_length:j+self.prediction_length]), torch.from_numpy(np.flip(ocean_run[j-self.prediction_length:j+self.prediction_length], 0).copy())
                j += 1
                i += 1
                
def generate_ocean_ds():
    train_ds = OceanToOcean(4, 'train')
    val_ds = OceanToOcean(4, 'valid')
    test_ds = OceanToOcean(4, 'test')

    return train_ds, val_ds, test_ds

class FluidToFluid(Dataset):

    def __init__(self, prediction_length, partition_name='train', fluid_val='u'):
        self.fluid_array = np.load(f'/g/data/x77/jm0124/fluids/{partition_name}_{fluid_val}.npy')
        self.prediction_length = prediction_length
    
    def __len__(self):
        return self.fluid_array.shape[0] - 2*self.prediction_length
    
    def __getitem__(self,idx):
        j = self.prediction_length
        for i in range(0,self.fluid_array.shape[0] - 2*self.prediction_length):
            if i == idx:
                array = self.fluid_array[j-self.prediction_length:j+self.prediction_length]
                reverse_array = np.flip(self.fluid_array[j-self.prediction_length:j+self.prediction_length]).copy()
                return torch.from_numpy(array), torch.from_numpy(reverse_array)
            j += 1

def generate_fluid_u():
    train_ds = FluidToFluid(4, 'train', 'u')
    val_ds = FluidToFluid(4, 'valid', 'u')
    test_ds = FluidToFluid(4, 'test', 'u')

    return train_ds, val_ds, test_ds

class PendulumToPendulum(Dataset):
    def __init__(self, prediction_length, dissipation_level, partition_name='train'):
        self.pendulum_array = np.load(f"/g/data/x77/jm0124/synthetic_datasets/pendulum_dissipative_{partition_name}.npy")
        self.dissipation_level = dissipation_level
        self.prediction_length = prediction_length
    
    def __len__(self):
        return len(self.pendulum_array[self.dissipation_level][:500])
    
    def __getitem__(self, idx):
        i = 0
        for pend_run in self.pendulum_array[self.dissipation_level][:500]:
            j = self.prediction_length
            for time_step in pend_run[self.prediction_length:-self.prediction_length]:
                if i == idx:
                    return torch.from_numpy(pend_run[j-self.prediction_length:j+self.prediction_length]), torch.from_numpy(np.flip(pend_run[j-self.prediction_length:j+self.prediction_length], 0).copy())
                j += 1
                i += 1

def generate_pendulum_ds(dissipation_level):
    train_ds = PendulumToPendulum(4, dissipation_level, 'train')
    val_ds = PendulumToPendulum(4, dissipation_level, 'valid')
    test_ds = PendulumToPendulum(4, dissipation_level, 'test')

    return train_ds, val_ds, test_ds

class LimitedDs(Dataset):
    def __init__(self, other_ds, length):
       self.ds = other_ds
       self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        return self.ds[idx]

def generate_limited_cyclones():
    train_ds, val_ds, test_ds = generate_example_dataset()

    a = LimitedDs(other_ds=train_ds, length=1000)

    return LimitedDs(other_ds=train_ds, length=1000), LimitedDs(other_ds=val_ds, length=1000), LimitedDs(other_ds=test_ds, length=1000)

def generate_example_dataset():
    """
    cyclone_dir, json_path, prediction_length, atmospheric_values, pressure_levels,
                load_np=True, save_np=False, partition_name='train', crop=True
    """

    train_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/train/', train_json_path, 4,
                                        ['u'], [2], load_np=True, save_np=False, partition_name='train')
    val_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/valid/', valid_json_path, 4,
                                        ['u'], [2], load_np=True, save_np=False, partition_name='valid')
    test_ds = CycloneToCycloneDataset('/g/data/x77/ob2720/partition/test/', test_json_path, 4,
                                        ['u'], [2], load_np=True, save_np=False, partition_name='test')

    return train_ds, val_ds, test_ds

def generate_prediction_dataset():
    train_ds = CycloneDataset('/g/data/x77/ob2720/partition/train/', tracks_path=train_json_path, 
                                        save_np=False, load_np=True, partition_name='train')
    val_ds = CycloneDataset('/g/data/x77/ob2720/partition/valid/', tracks_path=valid_json_path, 
                                        save_np=False, load_np=True, partition_name='valid')
    test_ds = CycloneDataset('/g/data/x77/ob2720/partition/test/', tracks_path=test_json_path,
                                         save_np=False, load_np=True, partition_name='test')
    
    print(len(train_ds))

    return train_ds, val_ds, test_ds

def generate_numpy_dataset(prediction_length, atmospheric_values, pressure_levels):
    """
    cyclone_dir, tracks_dict, save_np = False, load_np = True, crop = True, prediction_length = 0,
                atmospheric_values = ['u'], pressure_levels=[2], partition_name='train'
    """

    train_ds = CycloneDataset('/g/data/x77/ob2720/partition/train/', tracks_path=train_json_path, 
                                        save_np=True, load_np=False)
    val_ds = CycloneDataset('/g/data/x77/ob2720/partition/valid/', tracks_path=valid_json_path, 
                                        save_np=True, load_np=False, partition_name='valid')
    test_ds = CycloneDataset('/g/data/x77/ob2720/partition/test/', tracks_path=test_json_path,
                                         save_np=True, load_np=False, partition_name='test')
    print(len(train_ds))
    print("Train ds")
    for i,(cyclone_array, cyclone, j) in tqdm(enumerate(train_ds)):
        np.save(f'/g/data/x77/jm0124/np_cyclones_crop/{prediction_length}/u/{pressure_levels[0]}/train/{cyclone}-{j}.npy', cyclone_array)
    
    # print(len(val_ds))
    # print("Val ds")
    # for i,(cyclone_array, cyclone, j) in tqdm(enumerate(val_ds)):
    #     np.save(f'/g/data/x77/jm0124/np_cyclones_crop/{prediction_length}/u/{pressure_levels[0]}/valid/{cyclone}-{j}.npy', cyclone_array)

    print(len(test_ds))
    print("Test ds")
    for i,(cyclone_array, cyclone, j) in tqdm(enumerate(test_ds)):
        np.save(f'/g/data/x77/jm0124/np_cyclones_crop/{prediction_length}/u/{pressure_levels[0]}/test/{cyclone}-{j}.npy', cyclone_array)

if __name__ == '__main__':
    train_ds, val_ds, test_ds = generate_pendulum_ds(0)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=64, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=8, pin_memory=True, shuffle=True)
    img, output = next(iter(loader))
    print(img.shape)
