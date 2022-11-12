from datasets import *
from models import *
from tqdm import tqdm
from dl_pipeline import *
import matplotlib.pyplot as plt
#import seaborn
import logging
import numpy as np
import time
import argparse
import logging
#from wandb import wandb
import json
import os
from pathlib import Path
import dataset_generation
import csv
import time
import datetime

direct = os.getcwd()
if direct[10:16] == 'jm0124':
    saved_models_path = '/home/156/jm0124/kae-cyclones/saved_models'
else:
    saved_models_path = '/home/156/cn1951/kae-cyclones/saved_models'

<<<<<<< HEAD
def train(model, device, train_loader, val_loader, train_size, val_size, learning_rate, eigenLoss, epochs, eigenlossAlpha, weight_decay=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
=======
def train(model, device, train_loader, val_loader, train_size, val_size, learning_rate, eigenLoss, epochs, eigenlossAlpha, approx=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
>>>>>>> aacfec355fb921a8bbae364b3b637eb5f411ac3a
    criterion = nn.MSELoss().to(device)
    model.to(device)
    model.train()
    model.to(device)
    lamb, nu, eta = 1e1, 1, 1
    alpha = eigenlossAlpha
    loss_dict = {}
    eigvals, epoch_times = [], []
    start = time.time()

    for epoch in tqdm(range(epochs)):
    #for epoch in range(epochs):
        eigvals.append(torch.linalg.eigvals(model.dynamics.dynamics.weight))
        
        avg_loss, avg_fwd_loss, avg_bwd_loss, avg_iden_loss, avg_cons_loss, avg_eigen_loss = 0, 0, 0, 0, 0, 0
        
        for i, cyclone_array_list in enumerate(train_loader):
            closs, cfwd, cbwd, ciden, ccons, ceigen = 0, 0, 0, 0, 0, 0
            for data in cyclone_array_list:
                # [400, 10, 2]
                loss_bwd, loss_consist, loss_bwd, loss_consist = 0, 0, 0, 0
                cyclone_array = data[0].float().to(device)
                reversed_array = data[1].float().to(device)
                
                if approx:
                    mode_forward = 'forward-approx'
                    mode_back = 'backward-approx'
                    stepSpace = np.linspace(0, 4, 4, dtype=int)
                else:
                    mode_forward = 'forward'
                    mode_back = 'backward'
                    stepSpace = np.linspace(0, model.steps-2, model.steps-2, dtype=int)

                out, out_back = model(x=cyclone_array[0].unsqueeze(0).to(device), mode=mode_forward)

                for k in stepSpace:
                    if k == 0:
                        loss_fwd = criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))
                    else:
                        loss_fwd += criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))

                loss_identity = 0
                
                if model.back:
                    loss_identity += criterion(out[-1], cyclone_array[0].unsqueeze(0).to(device)) * model.steps
                    loss_bwd = 0.0
                    loss_consist = 0.0

                    loss_bwd = 0.0
                    loss_consist = 0.0

                    #print(f"Backward cyclone array {cyclone_array[-1].unsqueeze(0).to(device)}")
                    out, out_back = model(x=cyclone_array[-1].unsqueeze(0).to(device), mode=mode_back)

                    for k in stepSpace:
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

                    #loss += lamb * loss_identity +  nu * loss_bwd + eta * loss_consist
                    # ciden += lamb * loss_identity
                    # cbwd += nu * loss_bwd
                    # ccons += eta * loss_consist
                
                # print(loss_fwd)
                # print(loss_identity)
                # print(loss_bwd)
                # print(loss_consist)
                closs += loss_fwd + lamb * loss_identity + nu * loss_bwd + eta * loss_consist
                ciden += lamb * loss_identity
                cbwd += nu * loss_bwd
                ccons += eta * loss_consist
                cfwd += loss_fwd
                # print(model.dynamics.dynamics.weight)
                # print(model.backdynamics.dynamics.weight)

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
        
        now = str(datetime.datetime.now())

        if type(avg_bwd_loss) == int:
            if loss_dict == {}:
                loss_dict['loss'] = [avg_loss.cpu().item()/train_size]
                loss_dict['iden'] = [0]
                loss_dict['fwd'] = [avg_fwd_loss.cpu().item()/train_size]
                loss_dict['bwd'] = [avg_bwd_loss/train_size]
                loss_dict['cons'] = [avg_cons_loss/train_size]
                loss_dict['eigen'] = [avg_eigen_loss.cpu().item()/train_size]
                loss_dict['time'] = [now]
            else:
                loss_dict['loss'].append(avg_loss.cpu().item()/train_size)
                loss_dict['iden'].append(0)
                loss_dict['fwd'].append(avg_fwd_loss.cpu().item()/train_size)
                loss_dict['bwd'].append(avg_bwd_loss/train_size)
                loss_dict['cons'].append(avg_cons_loss/train_size)
                loss_dict['eigen'].append(avg_eigen_loss.cpu().item()/train_size)
                loss_dict['time'].append(now)
        else:
            if loss_dict == {}:
                loss_dict['loss'] = [avg_loss.cpu().item()/train_size]
                loss_dict['iden'] = [0]
                loss_dict['fwd'] = [avg_fwd_loss.cpu().item()/train_size]
                loss_dict['bwd'] = [avg_bwd_loss.cpu().item()/train_size]
                loss_dict['cons'] = [avg_cons_loss.cpu().item()/train_size]
                loss_dict['eigen'] = [avg_eigen_loss.cpu().item()/train_size]
                loss_dict['time'] = [now]
            else:
                loss_dict['loss'].append(avg_loss.cpu().item()/train_size)
                loss_dict['iden'].append(0)
                loss_dict['fwd'].append(avg_fwd_loss.cpu().item()/train_size)
                loss_dict['bwd'].append(avg_bwd_loss.cpu().item()/train_size)
                loss_dict['cons'].append(avg_cons_loss.cpu().item()/train_size)
                loss_dict['eigen'].append(avg_eigen_loss.cpu().item()/train_size)
                loss_dict['time'].append(now)

        w = torch.linalg.eigvals(model.dynamics.dynamics.weight)
            
        forward_val = eval_models(model, val_loader, val_size)

        if epoch == 0:
            loss_dict['fwd_val'] = [forward_val]
        else:
            loss_dict['fwd_val'].append(forward_val)
        end = time.time()
        epoch_times.append(end-start)
    
    return eigvals, loss_dict, epoch_times

def test_accuracy(model, device, test_loader, step_legnth):
    criterion = nn.MSELoss().to(device)
    model.eval()
    model.steps = step_legnth
    errors = []
    for i, cyclone_array_list in enumerate(test_loader):
        error_sequence = np.zeros(model.steps-1)
        closs, cfwd, cbwd, ciden, ccons, ceigen = 0, 0, 0, 0, 0, 0

        for data in cyclone_array_list:
            cyclone_array = data[0].float().to(device)
            reversed_array = data[1].float().to(device)

            out, out_back = model(x=cyclone_array[0].unsqueeze(0).to(device), mode='forward')
            
            for k in range(model.steps-1):
                error_sequence[k] = criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))
        
        errors.append(error_sequence)
    
    errors = np.array(errors)
    
    return np.average(errors, axis=0), np.std(errors, axis=0) 
    
def create_model(alpha, beta, init_scheme, input_size):
    "Creates a model after instantiating a dataset."
    model_dae = koopmanAE(init_scheme, b=beta, alpha=alpha, input_size=input_size).to(0)
    return model_dae

def create_dataset(dataset:str, batch_size):
    #print(dataset)
    "Build a dataset based on the problem type."
    if dataset.startswith('cyclone'):
        if dataset == 'cyclone': train_ds, val_ds, test_ds = generate_example_dataset()
        elif dataset == 'cyclone-limited':
            train_ds, val_ds, test_ds, test_steps = generate_limited_cyclones()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 400
        alpha = 16
<<<<<<< HEAD
        beta = 16
        learning_rate = 5e-4
        eigenlossHyper = 2e2

    elif dataset == 'pendulum0-200':
        train_ds, val_ds, test_ds, test_steps = generate_pendulum_ds(0, 200)

=======
        beta = 8
        learning_rate = 1e-3
        eigenlossHyper = 4e3
>>>>>>> aacfec355fb921a8bbae364b3b637eb5f411ac3a
    elif dataset == 'pendulum0':
        train_ds, val_ds, test_ds = generate_pendulum_ds(0)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 16
        learning_rate = 1e-4
        eigenlossHyper = 1e2
        
    elif dataset == 'pendulum0-100':
        train_ds, val_ds, test_ds, test_steps = generate_pendulum_ds(0, 26)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 16
        learning_rate = 1e-4
        eigenlossHyper = 1e2
    
    elif dataset == 'pendulum0-64':
        train_ds, val_ds, test_ds, test_steps = generate_pendulum_ds(0, 64)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 16
        learning_rate = 1e-4
        eigenlossHyper = 1e2
    
    elif dataset == 'pendulum0-30':
        train_ds, val_ds, test_ds, test_steps = generate_pendulum_ds(0, 30)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 16
        learning_rate = 1e-4
        eigenlossHyper = 3e1
    
    elif dataset == 'pendulum0-20':
        train_ds, val_ds, test_ds, test_steps = generate_pendulum_ds(0, 20)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 16
        learning_rate = 1e-5
        eigenlossHyper = 1e2
        
    elif dataset == 'fp-100':
        batch_size = 128
        train_ds, val_ds, test_ds, test_steps = generate_fpTofp_ds(5)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 8
        beta = 2
        learning_rate = 1e-4
        eigenlossHyper = 1e2
        
    elif dataset == 'duffing-100':
        batch_size = 1024
        train_ds, val_ds, test_ds, test_steps = generate_duffingToduffing_ds(100)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 16
        beta = 4
        learning_rate = 1e-5
        eigenlossHyper = 1e6
        
    elif dataset == 'pendulum3':
        train_ds, val_ds, test_ds = generate_pendulum_ds(3)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 4
        learning_rate = 1e-5
        eigenlossHyper = 5e1
    
    elif dataset == 'pendulum5':
        train_ds, val_ds, test_ds = generate_pendulum_ds(5)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 2
        alpha = 4
        beta = 4
        learning_rate = 1e-5
        eigenlossHyper = 5e1
    
    elif dataset == 'pendulum7':
        train_ds, val_ds, test_ds = generate_pendulum_ds(7)
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
        train_ds, val_ds, test_ds, test_steps  = generate_ocean_ds()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 150
        alpha = 32
        beta = 16
        learning_rate = 1e-4
        eigenlossHyper = 2
        
    elif dataset == 'fluid':
        train_ds, val_ds, test_ds, test_steps = generate_fluid_u()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        input_size = 89351
        alpha = 16
        beta = 16
        learning_rate = 1e-3
        eigenlossHyper = 10

    return train_ds, val_ds, test_ds, loader, val_loader, test_loader, test_steps, input_size, alpha, beta, learning_rate, eigenlossHyper