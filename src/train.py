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

direct = os.getcwd()
if direct[10:16] == 'jm0124':
    saved_models_path = '/home/156/jm0124/kae-cyclones/saved_models'
else:
    saved_models_path = '/home/156/cn1951/kae-cyclones/saved_models'

def train(model, device, train_loader, val_loader, train_size, val_size, learning_rate, eigenLoss, epochs, eigenlossAlpha):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)
    model.to(device)
    model.train()
    model.to(device)
    lamb, nu, eta = 1, 1, 1e-2
    alpha = eigenlossAlpha
    loss_dict = {}

    print(model.encoder.fc1)

    for epoch in range(epochs):
        avg_loss, avg_fwd_loss, avg_bwd_loss, avg_iden_loss, avg_cons_loss, avg_eigen_loss = 0, 0, 0, 0, 0, 0
        
        for i, cyclone_array_list in enumerate(train_loader):
            closs, cfwd, cbwd, ciden, ccons, ceigen = 0, 0, 0, 0, 0, 0
            
            for data in cyclone_array_list:
                cyclone_array = data[0].float().to(device)
                reversed_array = data[1].float().to(device)

                out, out_back = model(x=cyclone_array[0].unsqueeze(0).to(device), mode='forward')

                for k in range(model.steps - 1):
                    if k == 0:
                        loss_fwd = criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))
                    else:
                        loss_fwd += criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))

                loss_identity = criterion(out[-1], cyclone_array[0].unsqueeze(0).to(device)) * model.steps

                loss_bwd, loss_consist, loss_bwd, loss_consist = 0, 0, 0, 0
                    
                closs += loss_fwd + lamb * loss_identity + nu * loss_bwd + eta * loss_consist
                ciden += lamb * loss_identity
                cbwd += nu * loss_bwd
                ccons += eta * loss_consist
                cfwd += loss_fwd

                
                #A = model.dynamics.dynamics.weight.cpu().detach().numpy()
                #w, _ = np.linalg.eig(A)
                w = torch.linalg.eigvals(model.dynamics.dynamics.weight)
                if eigenLoss == 'origin_mse': w_pen = torch.nn.L1Loss()(torch.abs(w), torch.zeros(w.shape).to(w.device))
                elif eigenLoss == 'unit_circle_mae': w_pen = torch.nn.L1Loss()(torch.abs(w), torch.ones(w.shape).to(w.device))
                elif eigenLoss == 'unit_circle_mse': w_pen = torch.nn.MSELoss()(torch.abs(w), torch.ones(w.shape).to(w.device))
                else: w_pen = torch.tensor(1)
                closs += alpha * w_pen
                ceigen += alpha * w_pen
        
            optimizer.zero_grad(set_to_none=True)
            closs.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient clip
            optimizer.step()
        
            avg_loss += closs
            avg_iden_loss += ciden
            avg_fwd_loss += cfwd
            avg_bwd_loss += cbwd
            avg_cons_loss += ccons
            avg_eigen_loss += ceigen

        if loss_dict == {}:
            loss_dict['loss'] = [avg_loss.cpu().item()/train_size]
            loss_dict['iden'] = [avg_iden_loss.cpu().item()/train_size]
            loss_dict['fwd'] = [avg_fwd_loss.cpu().item()/train_size]
            loss_dict['bwd'] = [avg_bwd_loss/train_size]
            loss_dict['cons'] = [avg_cons_loss/train_size]
            loss_dict['eigen'] = [avg_eigen_loss.cpu().item()/train_size]
        else:
            loss_dict['loss'].append(avg_loss.cpu().item()/train_size)
            loss_dict['iden'].append(avg_iden_loss.cpu().item()/train_size)
            loss_dict['fwd'].append(avg_fwd_loss.cpu().item()/train_size)
            loss_dict['bwd'].append(avg_bwd_loss/train_size)
            loss_dict['cons'].append(avg_cons_loss/train_size)
            loss_dict['eigen'].append(avg_eigen_loss.cpu().item()/train_size)

        w = torch.linalg.eigvals(model.dynamics.dynamics.weight)
        # print(w)
        # print(f"Eigenloss: {avg_eigen_loss/train_size}")
            
        forward_val = eval_models(model, val_loader, val_size, koopman=True)[0][0]

        if epoch == 0:
            loss_dict['fwd_val'] = [forward_val]
        else:
            loss_dict['fwd_val'].append(forward_val)
    
    return loss_dict

def create_model(alpha, beta, init_scheme, input_size):
    "Creates a model after instantiating a dataset."
    model_dae = koopmanAE(init_scheme, b=beta, alpha=alpha, input_size=input_size).to(0)
    return model_dae

def create_dataset(dataset:str, batch_size):
    print(dataset)
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
        eigenlossHyper = 5e2

<<<<<<< HEAD

    elif dataset == 'pendulum':
        train_ds, val_ds, test_ds = generate_pendulum_ds(0)

=======
>>>>>>> 89e1e2c6e4082bcd65896a04ed908e9c90da2546
    elif dataset == 'pendulum0':
        train_ds, val_ds, test_ds = generate_pendulum_ds(0)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 4
        learning_rate = 1e-5
        eigenlossHyper = 1e2
    
    elif dataset == 'pendulum5':
        train_ds, val_ds, test_ds = generate_pendulum_ds(5)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 4
        learning_rate = 1e-5
        eigenlossHyper = 5e1
    
    elif dataset == 'pendulum9':
        train_ds, val_ds, test_ds = generate_pendulum_ds(9)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 4
        learning_rate = 1e-5
        eigenlossHyper = 5e1
            
    elif dataset == 'ocean':
        train_ds, val_ds, test_ds = generate_ocean_ds()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 150
        alpha = 16
        beta = 16
        learning_rate = 1e-4
        eigenlossHyper = 2
        
    elif dataset == 'fluid':
        train_ds, val_ds, test_ds = generate_fluid_u()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 89351
        alpha = 64
        beta = 16
        learning_rate = 1e-4
        eigenlossHyper = 10

    return train_ds, val_ds, test_ds, loader, val_loader, input_size, alpha, beta, learning_rate, eigenlossHyper