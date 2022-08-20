from datasets import *
from models import *
import dl_pipeline
from tqdm import tqdm
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

logging.basicConfig(level=logging.DEBUG, filename='log-ae-3.txt')
direct = os.getcwd()
if direct[10:16] == 'jm0124':
    saved_models_path = '/home/156/jm0124/kae-cyclones/saved_models'
else:
    saved_models_path = '/home/156/cn1951/kae-cyclones/saved_models'
print(f"Saved models path: {saved_models_path}")

def train(model, device, train_loader, val_loader, train_size, val_size, learning_rate, eigenvalue_penalty_type, eigen_dist, eigen_penalty, group, run="1", epochs=args.num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)
    model.train()
    
    loss_dict = {}

    for epoch in range(epochs):
        avg_loss, avg_fwd_loss, avg_bwd_loss, avg_iden_loss, avg_cons_loss, avg_eigen_loss = 0, 0, 0, 0, 0, 0
        
        for i, cyclone_array_list in enumerate(train_loader):
            closs, cfwd, cbwd, ciden, ccons, ceigen = 0, 0, 0, 0, 0, 0
            
            for data in cyclone_array_list:
                cyclone_array = data[0].float().to(device)
                reversed_array = data[1].float().to(device)

                out, out_back = model(x=cyclone_array[0].unsqueeze(0), mode='forward')

                for k in range(args.forward_steps - 1):
                    if k == 0:
                        loss_fwd = criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))
                    else:
                        loss_fwd += criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))

                loss_identity = criterion(out[-1], cyclone_array[0].unsqueeze(0).to(device)) * args.forward_steps

                loss_bwd, loss_consist, loss_bwd, loss_consist = 0, 0, 0, 0
                if args.model == 'constrainedKAE':
                    out, out_back = model(x=cyclone_array[-1].unsqueeze(0), mode='backward')

                    for k in range(args.forward_steps-1):
                        if k == 0:
                            loss_bwd = criterion(out_back[k], reversed_array[k+1].unsqueeze(0).to(device))
                        else:
                            loss_bwd += criterion(out_back[k], reversed_array[k+1].unsqueeze(0).to(device))

                    A = model.dynamics.dynamics.weight
                    B = model.backdynamics.dynamics.weight

                    K = A.shape[-1]

                    for k in range(1,K+1):
                        As1 = A[:,:k]
                        Bs1 = B[:k,:]
                        As2 = A[:k,:]
                        Bs2 = B[:,:k]

                        Ik = torch.eye(k).float().to(device)

                        if k == 1:
                            loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                            torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                        else:
                            loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                            torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)
                    
                closs += loss_fwd + args.lamb * loss_identity + args.nu * loss_bwd + args.eta * loss_consist
                ciden += args.lamb * loss_identity
                cbwd += args.nu * loss_bwd
                ccons += args.eta * loss_consist
                cfwd += loss_fwd

                if eigen_penalty:
                    A = model.dynamics.dynamics.weight.cpu().detach().numpy()
                    w, v = np.linalg.eig(A)
                    if eigenvalue_penalty_type == 'max':
                        w_pen = np.max(np.absolute(w))
                    elif eigenvalue_penalty_type == 'average':
                        w_pen = np.average(np.absolute(w))
                    elif eigenvalue_penalty_type == 'inverse':
                        w_pen = 1/np.min(np.absolute(w))
                    elif args.eigenvalue_penalty_type == 'unit_circle':
                        w_pen = np.sum(np.absolute(np.diff(1, w)))
                    closs += args.alpha * w_pen
                    ceigen += args.alpha * w_pen
        
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
            loss_dict['loss'] = [avg_loss/train_size]
            loss_dict['iden'] = [avg_iden_loss/train_size]
            loss_dict['fwd'] = [avg_fwd_loss/train_size]
            loss_dict['bwd'] = [avg_bwd_loss/train_size]
            loss_dict['cons'] = [avg_cons_loss/train_size]
            loss_dict['eigen'] = [avg_eigen_loss/train_size]
        else:
            loss_dict['loss'].append(avg_loss/train_size)
            loss_dict['iden'].append(avg_iden_loss/train_size)
            loss_dict['fwd'].append(avg_fwd_loss/train_size)
            loss_dict['bwd'].append(avg_bwd_loss/train_size)
            loss_dict['cons'].append(avg_cons_loss/train_size)
            loss_dict['eigen'].append(avg_eigen_loss/train_size)

        forward_val = dl_pipeline.eval_models(model, val_loader, val_size, koopman=True)[0][0]

        if epoch == 0:
            loss_dict['fwd_val'] = [forward_val]
        else:
            loss_dict['fwd_val'].append(forward_val)
    
        logging.info(loss_dict)
    
    return loss_dict

def create_model(alpha, beta, init_scheme, input_size):
    "Creates a model after instantiating a dataset."
    model_dae = koopmanAE(init_scheme, b=beta, alpha=alpha, input_size=input_size).to(0)
    return model_dae

def create_dataset(dataset:str):
    "Build a dataset based on the problem type."
    if dataset.startswith('cyclone'):
        if dataset == 'cyclone': train_ds, val_ds, test_ds = generate_example_dataset()
        elif dataset == 'cyclone-limited':
            train_ds, val_ds, test_ds = generate_limited_cyclones()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 400
        alpha = 16
        beta = 16
        learning_rate = 1e-3

    elif args.dataset == 'pendulum':
        train_ds, val_ds, test_ds = generate_pendulum_ds(args.dissipative_pendulum_level)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 4
        learning_rate = 1e-5
            
    elif args.dataset == 'ocean':
        train_ds, val_ds, test_ds = generate_ocean_ds()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 150
        alpha = 16
        beta = 16
        learning_rate = 1e-4
        
    elif args.dataset == 'fluid':
        train_ds, val_ds, test_ds = generate_fluid_u()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 89351
        alpha = 64
        beta = 16
        learning_rate = 1e-4

    return loader, val_loader, input_size, alpha, beta, learning_rate