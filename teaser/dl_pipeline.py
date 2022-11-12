from datasets import *
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import logging

# logging.basicConfig(level=logging.DEBUG, filename='log.txt')
# logging.debug('This will get logged')

def eval_models(model,  train_loader, ds_length, koopman=True, device=0, num_epochs=1, steps=4, lamb=1, nu=1, eta=1e-2, batch_size=16, backward=1):
    criterion = nn.MSELoss().to(device)
    model.eval()

    avg_fwd_loss = 0
    for i, cyclone_array_list in enumerate(train_loader):
        loss, cfwd, cbwd, ciden, ccons = 0, 0, 0, 0, 0                    
        for data in cyclone_array_list:
            cyclone_array = data[0].float() 
            cyclone_array = cyclone_array.to(device)
            
            out, out_back = model(x=cyclone_array[0].unsqueeze(0), mode='forward')

            loss_fwd = 0
            for k in range(model.steps-1):
                loss_fwd += criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))

            cfwd += loss_fwd

        avg_fwd_loss += cfwd.item()            

    return avg_fwd_loss/(ds_length)

def eval_ae_kae(model_kae, model_ae, val_ds, ds_length):
    print(f"KAE: {eval(model_kae, val_ds, ds_length)}")
    print(f"AE: {eval(model_ae, val_ds, ds_length)}")
            

def get_eigendecomp(model):
    tensor_array = model.dynamics.dynamics.weight
    koopman_operator = tensor_array.cpu().detach().numpy()

    w, v = np.linalg.eig(koopman_operator)

    return w, v

def reconstruct_operator(w, v):
    R = np.linalg.inv(v)
    # create diagonal matrix from eigenvalues
    L = np.diag(w)
    # reconstruct the original matrix
    B = v.dot(L).dot(R)
    return B

def create_synthetic_example(model, x, w, v, w_index):
    w[w_index] = 1
    operator = reconstruct_operator(w,v)

    model = model.to(0)
    
    old_example = model(x.to(0))[0][1]

    new_model = model.to(0)
    new_model.to(0)

    new_model.dynamics.dynamics.weight = nn.Parameter(torch.tensor(operator).cpu().float())

    new_model.to(0)

    new_example = new_model(x.to(0))[0][0]

    return new_example, old_example

def import_models(load=True):
    model_kae = koopmanAE(16, 4, 4)
    model_kae.to(0)
    model_ae = regularAE(16,4,4)
    model_ae.to(0)

    if load:
        model_kae.load_state_dict(torch.load("./saved_models/ae-model-continued-3.1817206502951767.pt"))
        # model_ae.load_state_dict(torch.load("./saved_models/ae-model-continued-7869.197857755121.pt"))
    
    return model_kae, model_ae

if __name__ == '__main__':
    model_dae = koopmanAE(32, steps=4, steps_back=4, alpha=8).to(0)

    # model_kae.load_state_dict(torch.load('./saved_models/kae-model-continued-4.082315057579133.pt'))
    # model_dae.load_state_dict(torch.load('./saved_models/ae-model-continued-0.6384171716724326.pt'))
    # regular_ae.load_state_dict(torch.load(f'{saved_models_path}/ae-model-continued-0.948112045283501.pt'))

    dataset, val_ds, test_ds = generate_example_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, num_workers=8, pin_memory=True, shuffle=True)
    model_dae, losses2, fwd_loss2, back_loss2, iden_loss2, cons_loss2 = train(model_dae, loader, len(dataset), koopman=False, num_epochs=50)
    # logging.info("Training AE")
    # regular_ae, losses3, fwd_loss3, back_loss3, iden_loss3, cons_loss3 = train(regular_ae, loader, len(dataset), koopman=False, num_epochs=30)