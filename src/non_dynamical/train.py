import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import *
import warnings
warnings.filterwarnings("ignore")

mnist_data = datasets.FashionMNIST('data', train=True, transform=transforms.ToTensor())
mnist_data = list(mnist_data)[:4096*2]
mnist_val = datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor())
mnist_val = list(mnist_val)[:4096*2]
print(len(mnist_val), len(mnist_data))

def eigenloss_func(w, loss_type=None):
    if loss_type == "minimal_inverse":
        return 1/np.min(np.abs(w))
    if loss_type == "average":
        return np.average(np.abs(w))
    if loss_type == "maximum":
        return np.max(np.abs(w))
    else:
        return w

def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3, eigenloss=False, 
          print_results=True, eigenloss_f="average"):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5)
    train_loader = torch.utils.data.DataLoader(mnist_data, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
    outputs, forward_losses, val_losses, eigenlosses = [], [], [], []
    for epoch in range(num_epochs):
        
        for data in train_loader:
            img, _ = data
            recon = model(img)
            forward_loss = criterion(recon, img)
            forward_losses.append(forward_loss)
            loss = 0
            loss += forward_loss
            
            if eigenloss:
                A = model.linear[1].weight.detach().numpy()
                w, v = np.linalg.eig(A)
                # assume this is already defined
                w_abs = eigenloss_func(w, loss_type=eigenloss_f)
                # w_abs = np.max(np.absolute(w))
                eigen_loss = 0.1 * w_abs
                eigenlosses.append(eigen_loss)
                loss += eigen_loss
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        for data in val_loader:
            with torch.no_grad():
                img, _ = data
                recon = model(img)
                val_loss = criterion(recon, img)
                val_losses.append(val_loss)
            
        if print_results:
            print('Epoch:{}, Forward Loss:{:.4f}, Val Loss:{:.4f}'.format(epoch+1, float(forward_loss), float(val_loss)))
            #if eigenloss: print("Eigenloss:{:.4f}".format(float(eigen_loss)))
        outputs.append((epoch, img, recon),)
    if eigenloss: return outputs, forward_losses, val_losses, eigenlosses
    else: return outputs, forward_losses, val_losses


if __name__ == "__main__":
    max_epochs, b = 5, 12
    model_1 = Autoencoder(b=b)
    outputs_1, losses_1, val_losses_1 = train(model_1, num_epochs=max_epochs)