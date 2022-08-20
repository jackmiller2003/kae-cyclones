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
import os
from pathlib import Path
import dataset_generation
import csv

os.environ["WANDB_MODE"] = "offline"

logging.basicConfig(level=logging.DEBUG, filename='log-ae-3.txt')
direct = os.getcwd()
if direct[10:16] == 'jm0124':
    saved_models_path = '/home/156/jm0124/kae-cyclones/saved_models'
else:
    saved_models_path = '/home/156/cn1951/kae-cyclones/saved_models'
print(f"Saved models path: {saved_models_path}")
wandb_dir= f"{str(Path(os.path.dirname(os.path.abspath('__file__'))).parents[0])}/results"

# TRAINING ARGUMENTS
parser = argparse.ArgumentParser(description='Autoencoder Prediction')
#
parser.add_argument('--model', type=str, default='dynamicKAE', metavar='N', help='model')
#
parser.add_argument('--pre_trained', type=str, default='', help='name of pretrained model in saved_models')
#
parser.add_argument('--num_epochs', type=int, default='50', help='number of epochs to train for')
#
parser.add_argument('--batch_size', type=int, default='256', help='batch size')
#
parser.add_argument('--forward_steps', type=int, default='4', help='number of forward steps')
#
parser.add_argument('--loss_terms', type=str, default='', help='e: eigen loss')
#
parser.add_argument('--lamb', type=float, default='1', help='identity factor')
#
parser.add_argument('--nu', type=float, default='1', help='backward factor')
#
parser.add_argument('--eta', type=float, default='1e-2', help='consistency factor')
#
parser.add_argument('--alpha', type=float, default='10', help='eigen factor')
#
parser.add_argument('--learning_rate', type=float, default='1e-3', help='learning rate')
#
parser.add_argument('--weight_decay', type=float, default='0.01', help='weight decay')
#
parser.add_argument('--eigen_init', type=str, default='True', help='initialise eigenvalues close to unit circle')
#
parser.add_argument('--eigen_init_maxmin', type=float, default='2', help='maxmin value for uniform distribution')
#
parser.add_argument('--experiment_name', type=str, default='', help='experiment name')
#
parser.add_argument('--dataset', type=str, default='cyclone', help='dataset')
#
parser.add_argument('--init_distribution', type=str, default='uniform', help='eigenvalue initialisation distribution')
#
parser.add_argument('--dissipative_pendulum_level', type=int, default='0', help='level of pendulum dissipative element')
#
parser.add_argument('--eigenvalue_penalty_type', type=str, default='max', help='type of penalty (max, average and inverse)')
#
parser.add_argument('--runs', type=int, default='3', help='number of runs')
#
parser.add_argument('--list_of_penalty', nargs='+', help='Penalty list', required=False)
#
parser.add_argument('--list_of_init', nargs='+', help='Init list', required=False)

args = parser.parse_args()

if args.experiment_name == '':
    args.experiment_name = f"experiment_{args.model}_{args.loss_terms}"

def train(model, device, train_loader, val_loader, train_size, val_size, learning_rate, eigenvalue_penalty_type, eigen_dist, eigen_penalty, group, run="1", epochs=args.num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)
    model.train()
    
    loss_dict = {}
    
    if args.dataset == "cyclone":
        project_wandb = "Koopman-autoencoders"
    elif args.dataset == "ocean":
        project_wandb = "ocean"
    elif args.dataset == "pendulum":
        project_wandb = "pendulum"

    # wandb.init(
    #   # Set the project where this run will be logged
    #   project="Koopman-autoencoders", 
    #   # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #   name=f"{group}:{args.dataset}:{run}", 
    #   dir=wandb_dir,
    #   # Track hyperparameters and run metadata
    #   config={
    #   "learning_rate": args.learning_rate,
    #   "architecture": args.model,
    #   "dataset": args.dataset,
    #   "epochs": args.num_epochs,
    #   "weight_decay": args.weight_decay,
    #   })
    
    # print(wandb.run.settings.mode)

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
                    
                    # How does torch backprop this?
                    # You can write the same code in torch
                    # eigvals give you back the eigenvalues
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
        
        # wandb.log({'forward validation': forward_val})
    
        logging.info(loss_dict)

        # wandb.log({
        #     'loss':avg_loss/train_size,
        #     'identity loss': avg_iden_loss/train_size,
        #     'forward loss': avg_fwd_loss/train_size,
        #     'backward loss': avg_bwd_loss/train_size,
        #     'consistency loss': avg_cons_loss/train_size,
        #     'eigenvalue loss': avg_eigen_loss/train_size
        # })

        with open(f'{group}.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write multiple rows
            writer.writerow([run,eigenvalue_penalty_type,eigen_dist, avg_fwd_loss.item()/train_size, forward_val])

            f.close()
    
    return model, loss_dict

def ei_finder():
    # generate dataset and instantiate learner
    train_ds, val_ds, test_ds = generate_ocean_ds()
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
    input_size = 150
    alpha = 16
    beta = 16
    learning_rate = 1e-4
    eigen_init = True if args.eigen_init == 'True' else False
    # initialisation stds
    lr_1 = [10**(-x) for x in range(10)]
    lr_1.reverse()
    lr_2 = [x for x in 10**np.linspace(0, 2, 10)]
    stds = lr_1 + lr_2
    # model and train  
    plot_losses = []
    for std in tqdm(stds):
        model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=eigen_init, eigen_distribution="gaussian", maxmin=args.eigen_init_maxmin, input_size=input_size, std=std).to(0)
        model, loss_dict = train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, args.eigenvalue_penalty_type, args.init_distribution, args.experiment_name, 0)
        # get losses
        losses, loss_items = loss_dict["loss"], []
        for loss in losses: loss_items.append(loss.detach().cpu().item())
        plot_losses.append(np.average(loss_items))
    print(plot_losses)
    plt.plot(stds, plot_losses)
    plt.xlabel("Standard deviation of eigeninit")
    plt.ylabel("Average validation loss")
    plt.show()
    return plot_losses
    
    
#if __name__ == '__main__':
   # ei_finder()


if __name__ == '__main__':
    if args.model == 'dynamicKAE':
        if args.dataset.startswith('cyclone'):
            if args.dataset == 'cyclone':
                train_ds, val_ds, test_ds = generate_example_dataset()
            elif args.dataset == 'cyclone-limited':
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

        eigen_init = True if args.eigen_init == 'True' else False

        # print(f'eigen init {eigen_init}')

        # if not (args.pre_trained == ''):
        #     print(f"Loading model: {args.pre_trained}")
        #     logging.info(f"Loading model: {args.pre_trained}")
        #     model_dae.load_state_dict(torch.load(f'{saved_models_path}/{args.pre_trained}'))

        logging.info("Training DAE")

        #if args.list_of_std != None:
        if 1 == 2: # @jack I think you meant to include an argument in args for list_of_std
            for std in args.list_of_std:
                for i in tqdm(range(0,args.runs)):
                    model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=True, eigen_distribution='uniform', maxmin=std, input_size=input_size).to(0)
                    train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, 'no-penalty', f'uniform-{std}', True, args.experiment_name, i)
        elif args.list_of_penalty != None:
            for penalty in args.list_of_penalty:
                for init in args.list_of_init:
                    for i in tqdm(range(0,args.runs)):
                        model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=True, eigen_distribution=init, maxmin=args.eigen_init_maxmin, input_size=input_size).to(0)
                        train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, penalty, init, True, args.experiment_name, i)
                for i in tqdm(range(0,args.runs)):
                    model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=False, eigen_distribution='gaussian element', maxmin=args.eigen_init_maxmin, input_size=input_size).to(0)
                    train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, penalty, 'gaussian element', True, args.experiment_name, i)
            for init in args.list_of_init:
                for i in tqdm(range(0,args.runs)):
                    model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=True, eigen_distribution=init, maxmin=args.eigen_init_maxmin, input_size=input_size).to(0)
                    train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, 'no-penalty', init, False, args.experiment_name, i)
            for i in tqdm(range(0,args.runs)):
                model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=False, eigen_distribution='gaussian element', maxmin=args.eigen_init_maxmin, input_size=input_size).to(0)
                train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, 'no-penalty', 'gaussian element', False, args.experiment_name, i)
        else:
            for i in range(0,args.runs):
                # print(i)
                model_dae = koopmanAE(beta, steps=4, steps_back=4, alpha=alpha, eigen_init=eigen_init, eigen_distribution=args.init_distribution, maxmin=args.eigen_init_maxmin, input_size=input_size).to(0)
                train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds), learning_rate, args.eigenvalue_penalty_type, args.init_distribution, args.experiment_name, i)