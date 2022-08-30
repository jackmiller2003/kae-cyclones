# -*- coding: utf-8 -*-
from datasets import *
from models import *
# import dl_pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import logging
import numpy as np
import logging
from wandb import wandb
import os
from pathlib import Path
import dataset_generation
import csv
from train_autoencoders import *
    
def valley(lrs:list, losses:list):
    n = len(losses)
    max_start, max_end = 0,0

    # find the longest valley
    lds = [1]*n
    for i in range(1,n):
        for j in range(0,i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]
    
    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections/2)

    return float(lrs[idx]), (float(lrs[idx]), losses[idx])


def ei_finder(search_epochs=3):
    # generate dataset and instantiate learner
    train_ds, val_ds, test_ds = generate_ocean_ds()
    loader = torch.utils.data.DataLoader(train_ds, batch_size=256, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, num_workers=8, pin_memory=True, shuffle=True)
    input_size = 150
    alpha = 16
    beta = 16
    learning_rate = 1e-4
    eigen_init = True
    # initialisation stds
    lr_1 = [10**(-x) for x in range(10)]
    lr_1.reverse()
    lr_2 = [x for x in 10**np.linspace(0, 2, 10)]
    stds = lr_1 + lr_2
    # model and train  
    plot_losses = []
    for std in tqdm(stds):
        model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=eigen_init, eigen_distribution="gaussian", maxmin=args.eigen_init_maxmin, input_size=input_size, std=std).to(0)
        model, loss_dict = train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, "max", "gaussian", "ei_finder", 0, epochs=search_epochs)
        # get losses
        losses, loss_items = loss_dict["loss"], []
        for loss in losses: loss_items.append(loss.detach().cpu().item())
        plot_losses.append(np.average(loss_items))
    return plot_losses

def plot_ei_finder():
    plot_losses = ei_finder()
    val, idx = valley(stds, losses)
    plt.plot(stds, losses, '-o')
    plt.xscale('log')
    plt.plot(val, idx[1], 'o', label="Valley", c="orange")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    plot_losses = ei_finder()
    print(plot_losses)
