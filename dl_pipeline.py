from datasets import *
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import logging

logging.basicConfig(level=logging.DEBUG, filename='log.txt')
logging.debug('This will get logged')

def train(model, train_loader, ds_length, koopman=True, device=0, num_epochs=20, steps=4, lamb=1, nu=1, eta=1e-2, batch_size=128, backward=1):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)
    model.train()

    avg_loss = 0

    fwd_loss = []
    back_loss = []
    iden_loss = []
    cons_loss = []
    losses = []

    for epoch in range(num_epochs):  
        avg_loss, avg_fwd_loss, avg_bwd_loss, avg_iden_loss, avg_cons_loss = 0, 0, 0, 0, 0      
        for i, cyclone_array_list in tqdm(enumerate(train_loader), total=ds_length/batch_size):
            loss, cfwd, cbwd, ciden, ccons = 0, 0, 0, 0, 0
                    
            if i == 0:
                model.print_hidden = True
            if i != 0:
                model.print_hidden = False

            for data in cyclone_array_list:
                cyclone_array = data[0].float() 
                reversed_array = data[1].float()
                cyclone_array = cyclone_array.to(device)
                reversed_array = reversed_array.to(device)

                out, out_back = model(x=cyclone_array[0].unsqueeze(0), mode='forward')

                for k in range(steps-1):
                    if k == 0:
                        np.save('example-in.npy', out[k].cpu().detach().numpy())
                        np.save('example-out.npy', cyclone_array[k+1].cpu().detach().numpy())
                        loss_fwd = criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))
                    else:
                        loss_fwd += criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))

                cfwd += loss_fwd

                loss_identity = criterion(out[-1], cyclone_array[0].unsqueeze(0).to(device)) * steps
                
                if koopman:
                    loss_bwd = 0.0
                    loss_consist = 0.0

                    loss_bwd = 0.0
                    loss_consist = 0.0

                    if backward == 1:
                        out, out_back = model(x=cyclone_array[-1].unsqueeze(0), mode='backward')

                        for k in range(steps-1):
                            
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

                    loss += loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist
                    ciden += lamb * loss_identity
                    cbwd += nu * loss_bwd
                    ccons += eta * loss_consist
                
                else:
                    #loss += loss_identity 
                    loss += loss_fwd

            avg_fwd_loss += cfwd.item()
            if koopman:
                avg_iden_loss += ciden.item()
                avg_bwd_loss += cbwd.item()
                avg_cons_loss += ccons.item()
            else:
                avg_iden_loss, avg_bwd_loss, avg_cons_loss = 0, 0, 0
            
            avg_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clip
            optimizer.step()

            if i % 1000 == 999:
                print(f"Loss: {avg_loss / (i * batch_size)}")
                print(f"Fwd loss: {avg_fwd_loss / (i * batch_size)}")
                # print(f"Iden loss: {avg_iden_loss / (i * batch_size)}")
                # print(f"Back loss: {avg_bwd_loss / (i * batch_size)}")
                # print(f"Cons loss: {avg_cons_loss / (i * batch_size)}")

        fwd_loss.append(avg_fwd_loss/(ds_length))
        back_loss.append(avg_bwd_loss/(ds_length))
        iden_loss.append(avg_iden_loss/(ds_length))
        cons_loss.append(avg_cons_loss/(ds_length))
        losses.append(avg_loss / (ds_length))

        logging.info(f"{epoch}. {avg_loss/(ds_length)}")
        logging.info(f"Fwd loss: {fwd_loss}")
        logging.info(f"Back loss: {back_loss}")
        logging.info(f"Iden loss: {iden_loss}")
        logging.info(f"Cons loss: {cons_loss}")

        print(f"{epoch}. {avg_loss/(ds_length)}")
        print(f"Fwd loss: {fwd_loss}")
        print(f"Back loss: {back_loss}")
        print(f"Iden loss: {iden_loss}")
        print(f"Cons loss: {cons_loss}")
        
        if koopman:
            torch.save(model.state_dict(), f'./saved_models/kae-model-continued-{avg_loss/ds_length}.pt')
        else:
            torch.save(model.state_dict(), f'./saved_models/ae-model-continued-{avg_loss/ds_length}.pt')
        
    
    return model, losses, fwd_loss, back_loss, iden_loss, cons_loss

def eval_models(model,  train_loader, ds_length, koopman=True, device=0, num_epochs=1, steps=4, lamb=1, nu=1, eta=1e-2, batch_size=128, backward=1):
    criterion = nn.MSELoss().to(device)
    model.eval()
    avg_loss = 0

    fwd_loss = []
    back_loss = []
    iden_loss = []
    cons_loss = []
    losses = []

    for epoch in range(num_epochs):  
        avg_loss, avg_fwd_loss, avg_bwd_loss, avg_iden_loss, avg_cons_loss = 0, 0, 0, 0, 0      
        for i, cyclone_array_list in tqdm(enumerate(train_loader), total=ds_length/batch_size):
            loss, cfwd, cbwd, ciden, ccons = 0, 0, 0, 0, 0
                    
            if i == 0:
                model.print_hidden = True
            if i != 0:
                model.print_hidden = False

            for data in cyclone_array_list:
                cyclone_array = data[0].float() 
                reversed_array = data[1].float()
                cyclone_array = cyclone_array.to(device)
                reversed_array = reversed_array.to(device)

                out, out_back = model(x=cyclone_array[0].unsqueeze(0), mode='forward')

                for k in range(steps-1):
                    if k == 0:
                        np.save('example-in.npy', out[k].cpu().detach().numpy())
                        np.save('example-out.npy', cyclone_array[k+1].cpu().detach().numpy())
                        loss_fwd = criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))
                    else:
                        loss_fwd += criterion(out[k], cyclone_array[k+1].unsqueeze(0).to(device))

                cfwd += loss_fwd

                loss_identity = criterion(out[-1], cyclone_array[0].unsqueeze(0).to(device)) * steps
                
                if koopman:
                    loss_bwd = 0.0
                    loss_consist = 0.0

                    loss_bwd = 0.0
                    loss_consist = 0.0

                    if backward == 1:
                        out, out_back = model(x=cyclone_array[-1].unsqueeze(0), mode='backward')

                        for k in range(steps-1):
                            
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

                    loss += loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist
                    ciden += lamb * loss_identity
                    cbwd += nu * loss_bwd
                    ccons += eta * loss_consist
                
                else:
                    #loss += loss_identity 
                    loss += loss_fwd

            avg_fwd_loss += cfwd.item()
            if koopman:
                avg_iden_loss += ciden.item()
                avg_bwd_loss += cbwd.item()
                avg_cons_loss += ccons.item()
            else:
                avg_iden_loss, avg_bwd_loss, avg_cons_loss = 0, 0, 0
            
            avg_loss += loss.item()

            if i % 100 == 99:
                print(f"Loss: {avg_loss / (i * batch_size)}")
                print(f"Fwd loss: {avg_fwd_loss / (i * batch_size)}")
                # print(f"Iden loss: {avg_iden_loss / (i * batch_size)}")
                # print(f"Back loss: {avg_bwd_loss / (i * batch_size)}")
                # print(f"Cons loss: {avg_cons_loss / (i * batch_size)}")

        fwd_loss.append(avg_fwd_loss/(ds_length))
        back_loss.append(avg_bwd_loss/(ds_length))
        iden_loss.append(avg_iden_loss/(ds_length))
        cons_loss.append(avg_cons_loss/(ds_length))
        losses.append(avg_loss / (ds_length))


        print(f"{epoch}. {avg_loss/(ds_length)}")
        print(f"Fwd loss: {fwd_loss}")
        print(f"Back loss: {back_loss}")
        print(f"Iden loss: {iden_loss}")
        print(f"Cons loss: {cons_loss}")

        return fwd_loss, back_loss, iden_loss, cons_loss

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
    model_kae = koopmanAE(4, steps=4, steps_back=4, alpha=1).to(0)
    model_dae = koopmanAE(4, steps=4, steps_back=4, alpha=1).to(0)
    regular_ae = regularAE(4, steps=4, steps_back=4, alpha=1).to(0)

    model_kae.load_state_dict(torch.load('./saved_models/kae-model-continued-4.082315057579133.pt'))
    model_dae.load_state_dict(torch.load('./saved_models/ae-model-continued-0.6384171716724326.pt'))
    regular_ae.load_state_dict(torch.load('./saved_models/ae-model-continued-0.948112045283501.pt'))

    dataset, val_ds, test_ds = generate_example_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, num_workers=8, pin_memory=True, shuffle=True)

    logging.info("Training DAE")
    model_dae, losses2, fwd_loss2, back_loss2, iden_loss2, cons_loss2 = train(model_dae, loader, len(dataset), koopman=False, num_epochs=30)
    logging.info("Training AE")
    regular_ae, losses3, fwd_loss3, back_loss3, iden_loss3, cons_loss3 = train(regular_ae, loader, len(dataset), koopman=False, num_epochs=30)

    logging.info(f"{losses}, {fwd_loss}, {back_loss}, {iden_loss}, {cons_loss}")
    logging.info(f"{losses2}, {fwd_loss2}, {back_loss2}, {iden_loss2}, {cons_loss2}")
    logging.info(f"{losses3}, {fwd_loss3}, {back_loss3}, {iden_loss3}, {cons_loss3}")