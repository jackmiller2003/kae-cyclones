import torch
import xarray
from datasets import *
from models import *
from dl_pipeline import *
from loss import *
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import argparse

# TRAINING ARGUMENTS
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='predictionANN', metavar='N', help='model')
#
parser.add_argument('--num_epochs', type=int, default='8', help='number of epochs to train for')
#
parser.add_argument('--batch_size', type=int, default='256', help='batch size')

args = parser.parse_args()


base_train_ds = CycloneDataset('/g/data/x77/ob2720/partition/train/', tracks_path=train_json_path, 
                            save_np=False, load_np=True, partition_name='train', synthetic=True, 
                            synthetic_type='base_synthesis', sigma=0.1)
normal_perturb_train_ds = CycloneDataset('/g/data/x77/ob2720/partition/train/', tracks_path=train_json_path, 
                            save_np=False, load_np=True, partition_name='train', synthetic=True, 
                            synthetic_type='normal_perturb_synthesis', sigma=0.1)

base_loader = torch.utils.data.DataLoader(base_train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
normal_perturb_loader = torch.utils.data.DataLoader(normal_perturb_train_ds, batch_size=256, num_workers=8, pin_memory=True, shuffle=True)

prediction_model = predictionANN(1)

def train(model, train_loader, ds_length, num_epochs, batch_size):
    loss_fn = L2_Dist_Func_Mae().to(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        for i, data in tqdm(enumerate(train_loader), total = ds_length/batch_size):
            
            if data == []:
                continue
            else:
                example = data[0]
                label = data[1]
            
            pred = model.forward(example)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()     

        print(f"Average loss: {loss}")
    
    return model


train(prediction_model, base_loader, len(base_train_ds), num_epochs=args.num_epochs, batch_size=args.batch_size)