from datasets import *
from models import *
from tqdm import tqdm
from dl_pipeline import *
import matplotlib.pyplot as plt
import seaborn
import logging
import numpy as np
import argparse
import logging
from wandb import wandb
import json
import os
from pathlib import Path
import dataset_generation
import csv
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

### Setup ###

logging.basicConfig(level=logging.DEBUG, filename='log-ae-3.txt')
direct = os.getcwd()
if direct[10:16] == 'jm0124':
    saved_models_path = '/home/156/jm0124/kae-cyclones/saved_models'
else:
    saved_models_path = '/home/156/cn1951/kae-cyclones/saved_models'

def setup(rank, worldSize):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12352' # Function to try and connect the port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=worldSize) # Stick to nccl.

def cleanup():
    print(loss_dict)
    dist.destroy_process_group()

def prepare(rank, world_size, dataset, batch_size=256, pin_memory=False, num_workers=8):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=sampler)
    
    print(f"Length of dataloader is {len(dataloader)}")
    
    return dataloader

### End of Setup ###

def loader_cycle(ddp_model, loader, ds_size, loss_dict, trainFlag):
    if not trainFlag:
        ddp_model.eval()

    avg_loss, avg_fwd_loss, avg_bwd_loss, avg_iden_loss, avg_cons_loss, avg_eigen_loss = 0, 0, 0, 0, 0, 0
    
    for i, cyclone_array_list in enumerate(loader):
        closs, cfwd, cbwd, ciden, ccons, ceigen = 0, 0, 0, 0, 0, 0
        
        for data in cyclone_array_list:
            cyclone_array = data[0].float().to(device)
            reversed_array = data[1].float().to(device)

            out, out_back = ddp_model(x=cyclone_array[0].unsqueeze(0), mode='forward')

            for k in range(ddp_model.steps - 1):
                if k == 0:
                    loss_fwd = criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))
                else:
                    loss_fwd += criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))

            loss_identity = criterion(out[-1], cyclone_array[0].unsqueeze(0).to(device)) * ddp_model.steps

            loss_bwd, loss_consist, loss_bwd, loss_consist = 0, 0, 0, 0
                
            closs += loss_fwd + lamb * loss_identity + nu * loss_bwd + eta * loss_consist
            ciden += lamb * loss_identity
            cbwd += nu * loss_bwd
            ccons += eta * loss_consist
            cfwd += loss_fwd

            
            A = ddp_model.dynamics.dynamics.weight.cpu().detach().numpy()
            w, _ = np.linalg.eig(A)
            if eigenLoss == 'max': w_pen = np.max(np.absolute(w))
            elif eigenLoss == 'average': w_pen = np.average(np.absolute(w))
            elif eigenLoss == 'inverse': w_pen = 1/np.min(np.absolute(w))
            elif eigenLoss == 'unit_circle': w_pen = np.sum(np.absolute(np.diff(1, w)))
            else: w_pen = 0
            closs += alpha * w_pen
            ceigen += alpha * w_pen
        
        if trainFlag:
            optimizer.zero_grad(set_to_none=True)
            closs.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.5) # gradient clip
            optimizer.step()
    
        avg_loss += closs
        avg_iden_loss += ciden
        avg_fwd_loss += cfwd
        avg_bwd_loss += cbwd
        avg_cons_loss += ccons
        avg_eigen_loss += ceigen

    if loss_dict == {}:
        loss_dict['loss'] = [avg_loss/ds_size]
        loss_dict['iden'] = [avg_iden_loss/ds_size]
        loss_dict['fwd'] = [avg_fwd_loss/ds_size]
        loss_dict['bwd'] = [avg_bwd_loss/ds_size]
        loss_dict['cons'] = [avg_cons_loss/ds_size]
        loss_dict['eigen'] = [avg_eigen_loss/ds_size]
    else:
        loss_dict['loss'].append(avg_loss/ds_size)
        loss_dict['iden'].append(avg_iden_loss/ds_size)
        loss_dict['fwd'].append(avg_fwd_loss/ds_size)
        loss_dict['bwd'].append(avg_bwd_loss/ds_size)
        loss_dict['cons'].append(avg_cons_loss/ds_size)
        loss_dict['eigen'].append(avg_eigen_loss/ds_size)
    
    return loss_dict

def train(eigenInit, std, eigenLoss, datasetName, batchSize, epochs):
    train_ds, val_ds, _, train_loader, val_loader, input_size, alpha, beta, lr = create_dataset(datasetName, batchSize)
    init_scheme = InitScheme(eigenInit, std, beta)
    model = koopmanAE(init_scheme, beta, alpha, input_size)

    worldSize = torch.cuda.device_count()

    mp.spawn(train_instance,
             args=(worldSize, model, train_ds, val_ds, lr, eigenLoss, epochs),
             nprocs=worldSize,
             join=True)

    loss_dict = train(model, 0, train_ds, val_ds, len(train_ds), len(val_ds), lr, eigenLoss, epochs)

def train_instance(rank, worldSize, model, train_ds, val_ds, learning_rate, eigenLoss, epochs):
    
    train_size, val_size = len(train_ds), len(val_ds)
    train_loader = prepare(rank, world_size, dataset)
    val_loader = prepate(rank, world_size, val_ds)
    
    setup(rank, worldSize)
    model.to(rank)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)
    
    ddp_model.train()
    
    lamb, nu, eta, alpha = 1, 1, 1e-2, 10
    loss_dict = {}
    val_dict = {}

    for epoch in range(epochs):
        loss_dict = loader_cycle(ddp_model, train_loader, train_size, loss_dict)
        val_dict = loader_cycle(ddp_model, train_loader, train_size, val_dict, trainFlag=False)['fwd']

        if epoch == 0:
            loss_dict['fwd_val'] = [forward_val]
        else:
            loss_dict['fwd_val'].append(forward_val)
    
    loss_dict['fwd_val'] = val_dict['fwd']

    return loss_dict

def create_model(alpha, beta, init_scheme, input_size):
    "Creates a model after instantiating a dataset."
    model_dae = koopmanAE(init_scheme, b=beta, alpha=alpha, input_size=input_size).to(0)
    return model_dae

def create_dataset(dataset:str, batch_size):
    "Build a dataset based on the problem type."
    if dataset.startswith('cyclone'):
        if dataset == 'cyclone': train_ds, val_ds, test_ds = generate_example_dataset()
        elif dataset == 'cyclone-limited':
            train_ds, val_ds, test_ds = generate_limited_cyclones()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 400
        alpha = 16
        beta = 16
        learning_rate = 1e-3

    elif dataset == 'pendulum':
        train_ds, val_ds, test_ds = generate_pendulum_ds(args.dissipative_pendulum_level)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 4
        learning_rate = 1e-5
            
    elif dataset == 'ocean':
        train_ds, val_ds, test_ds = generate_ocean_ds()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 150
        alpha = 16
        beta = 16
        learning_rate = 1e-4
        
    elif dataset == 'fluid':
        train_ds, val_ds, test_ds = generate_fluid_u()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 89351
        alpha = 64
        beta = 16
        learning_rate = 1e-4

    return train_ds, val_ds, test_ds, loader, val_loader, input_size, alpha, beta, learning_rate