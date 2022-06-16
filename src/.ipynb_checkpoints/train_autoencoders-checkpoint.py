from datasets import *
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import logging
import numpy as np
import argparse
import logging
from wandb import wandb
import os
<<<<<<< HEAD
from pathlib import Path

os.environ["WANDB_MODE"] = "offline"

logging.basicConfig(level=logging.DEBUG, filename='log-ae-3.txt')
saved_models_path = '/home/156/jm0124/kae-cyclones/saved_models'
wandb_dir= f"{str(Path(os.path.dirname(os.path.abspath('__file__'))).parents[0])}/results"
=======

os.environ["WANDB_MODE"] = "offline"

logging.basicConfig(level=logging.DEBUG, filename='log-ae.txt')
saved_models_path = '/home/156/jm0124/kae-cyclones/saved_models'
>>>>>>> refs/remotes/origin/main

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
<<<<<<< HEAD
parser.add_argument('--weight_decay', type=float, default='0.01', help='learning rate')
#
parser.add_argument('--eigen_init', type=bool, default='True', help='initialise eigenvalues close to unit circle')
#
parser.add_argument('--experiment_name', type=str, default='', help='experiment name')

args = parser.parse_args()

if args.experiment_name == '':
    args.experiment_name = f"experiment_{args.model}_{args.loss_terms}"

=======
parser.add_argument('--weight_decay', type=float, default='0.01', help='weight decay')

args = parser.parse_args()

>>>>>>> refs/remotes/origin/main
def train(model, device, train_loader, val_loader, train_size, val_size):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)
    model.train()
    
    loss_dict = {}

    wandb.init(
      # Set the project where this run will be logged
      project="Koopman-autoencoders", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
<<<<<<< HEAD
      name=args.experiment_name, 
      dir=wandb_dir,
=======
      #name=f"experiment_{args.model}_{args.loss_terms}", 
      name = "eigenloss_sum_of_eigenvalues_wd=0.05",
>>>>>>> refs/remotes/origin/main
      # Track hyperparameters and run metadata
      config={
      "learning_rate": args.learning_rate,
      "architecture": args.model,
      "dataset": "Sub-sampled cyclone dataset",
      "epochs": args.num_epochs,
<<<<<<< HEAD
      "weight_decay": args.weight_decay,
=======
      "weight_decay": args.weight_decay
>>>>>>> refs/remotes/origin/main
      })
    
    print(wandb.run.settings.mode)

    for epoch in range(args.num_epochs):
        avg_loss, avg_fwd_loss, avg_bwd_loss, avg_iden_loss, avg_cons_loss, avg_eigen_loss = 0, 0, 0, 0, 0, 0
        
        for i, cyclone_array_list in tqdm(enumerate(train_loader), total = int(train_size/args.batch_size)):
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

                if args.loss_terms == 'e':
                    A = model.dynamics.dynamics.weight.cpu().detach().numpy()
                    w, v = np.linalg.eig(A)
<<<<<<< HEAD
                    w_abs = np.max(np.absolute(w))
=======
                    # w_abs = np.max(np.absolute(w))
                    # sum all eigenvalue magnitudes and divide by number of eigenvalues
                    w_abs = np.sum(np.absolute(w))/w.shape[0]
>>>>>>> refs/remotes/origin/main
                    closs += args.alpha * w_abs
                    ceigen += args.alpha * w_abs
        
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
        
<<<<<<< HEAD
        print("Logging in log-ae")
        logging.info(loss_dict)

        print("Logging wandb")

=======
>>>>>>> refs/remotes/origin/main
        wandb.log({
            'loss':avg_loss/train_size,
            'identity loss': avg_iden_loss/train_size,
            'forward loss': avg_fwd_loss/train_size,
            'backward loss': avg_bwd_loss/train_size,
            'consistency loss': avg_cons_loss/train_size,
            'eigenvalue loss': avg_eigen_loss/train_size
        })

        logging.info(loss_dict)
    
    torch.save(model.state_dict(), f'{saved_models_path}/dae-eigen-{avg_loss/train_size}.pt')
    
    return model, loss_dict

if __name__ == '__main__':
    if args.model == 'dynamicKAE':
        train_ds, val_ds, test_ds = generate_example_dataset()
        model_dae = koopmanAE(16, steps=4, steps_back=4, alpha=16).to(0)

        if not (args.pre_trained == ''):
            print(f"Loading model: {args.pre_trained}")
            logging.info(f"Loading model: {args.pre_trained}")
            model_dae.load_state_dict(torch.load(f'{saved_models_path}/{args.pre_trained}'))

        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)

        logging.info("Training DAE")
        train(model_dae, 0, loader, val_loader, len(train_ds), len(val_ds))