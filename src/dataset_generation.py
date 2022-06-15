# Copied from https://github.com/erichson/koopmanAE/blob/master/read_dataset.py

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0, theta=2.4):
    if name == 'pendulum_lin':
        return pendulum_lin(noise)      
    if name == 'pendulum':
        return pendulum(noise, theta)    
    else:
        raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test

def pendulum_lin(noise):
    
    np.random.seed(0)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    
    X = sol(anal_ts, 0.8)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
 
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate   
    Xclean = Xclean.T.dot(Q.T)     
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1


def pendulum(noise, theta=2.4):
    
    np.random.seed(1)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    X = sol(anal_ts, theta)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:1400]   
    X_val = X[1400:1800]
    X_test = X[1800:2200]

    X_train_clean = Xclean[0:1400]   
    X_val_clean = Xclean[1400:1800]
    X_test_clean = Xclean[1800:2200]  
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_val, X_test, X_train_clean, X_val_clean, X_test_clean, 64, 1

def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1],1)

    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    else:
        return "dimenional error"

def pendulum_to_ds(steps, batch_size):
    Xtrain, Xval, Xtest, Xtrain_clean, Xval_clean, Xtest_clean, m, n = pendulum(0)

    #******************************************************************************
    # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
    #******************************************************************************
    Xtrain = add_channels(Xtrain)
    Xval = add_channels(Xval)
    Xtest = add_channels(Xtest)

    # transfer to tensor
    Xtrain = torch.from_numpy(Xtrain).float().contiguous()
    Xval = torch.from_numpy(Xval).float().contiguous()
    Xtest = torch.from_numpy(Xtest).float().contiguous()

    #******************************************************************************
    # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
    #******************************************************************************
    Xtrain_clean = add_channels(Xtrain_clean)
    Xval_clean = add_channels(Xval_clean)
    Xtest_clean = add_channels(Xtest_clean)

    # transfer to tensor
    Xtrain_clean = torch.from_numpy(Xtrain_clean).float().contiguous()
    Xval_clean = torch.from_numpy(Xval_clean).float().contiguous()
    Xtest_clean = torch.from_numpy(Xtest_clean).float().contiguous()

    #******************************************************************************
    # Create Dataloader objects
    #******************************************************************************
    trainDat = []
    start = 0
    for i in np.arange(steps,-1, -1):
        if i == 0:
            trainDat.append(Xtrain[start:].float())
        else:
            trainDat.append(Xtrain[start:-i].float())
        start += 1

    train_data = torch.utils.data.TensorDataset(*trainDat)
    del(trainDat)

    valDat = []
    start = 0
    for i in np.arange(steps,-1, -1):
        if i == 0:
            valDat.append(Xval[start:].float())
        else:
            valDat.append(Xval[start:-i].float())
        start += 1

    val_data = torch.utils.data.TensorDataset(*valDat)
    del(valDat)

    testDat = []
    start = 0
    for i in np.arange(steps,-1, -1):
        if i == 0:
            testDat.append(Xtest[start:].float())
        else:
            testDat.append(Xtest[start:-i].float())
        start += 1

    test_data = torch.utils.data.TensorDataset(*testDat)
    del(testDat)

    return train_data, val_data, test_data

if __name__ == '__main__':
    print(pendulum_to_ds(4,64)[0])