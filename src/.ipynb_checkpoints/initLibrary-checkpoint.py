import torch
import numpy as np

def reconstruct_operator(w, v):
    """
    Recreate a matrix from its eigenvalues and eigenvectors.
    """
    R = np.linalg.inv(v)
    # create diagonal matrix from eigenvalues
    L = np.diag(w)
    # reconstruct the original matrix
    B = v.dot(L).dot(R)
    return B

def gaussianElement(std, matrixSize):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/matrixSize]))
    Omega = sampler.sample((matrixSize, matrixSize))[..., 0]
    return Omega

def gaussianEigen(std, matrixSize):
    Omega = gaussianElement(std, matrixSize)
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())
    w.real = np.random.uniform(-std,std, w.shape[0])
    imag_dist = np.random.uniform(-std,std, w.shape[0])
    w = w + np.zeros(w.shape[0], dtype=complex)
    w.imag = imag_dist
    return torch.from_numpy(reconstruct_operator(w,v).real).float()

def doubleGaussianEigen(std, matrixSize):
    Omega = gaussianElement(std, matrixSize)
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())

    w.real = np.random.normal(loc=1, scale=std, size=w.shape[0]) + np.random.normal(loc=-1, scale=std, size=w.shape[0])
    imag_dist = np.random.normal(loc=1, scale=std, size=w.shape[0]) + np.random.normal(loc=-1, scale=std, size=w.shape[0])
    w = w + np.zeros(w.shape[0], dtype=complex)
    w.imag = imag_dist
    
    return torch.from_numpy(reconstruct_operator(w,v).real).float()

def uniformEigen(std, matrixSize):
    Omega = gaussianElement(std, matrixSize)
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())

    w.real = np.random.uniform(-std,std, w.shape[0])
    imag_dist = np.random.uniform(-std,std, w.shape[0])
    w = w + np.zeros(w.shape[0], dtype=complex)
    w.imag = imag_dist
    
    return torch.from_numpy(reconstruct_operator(w,v).real).float()

def svdElement(std, matrixSize):
    Omega = gaussianElement(std, matrixSize)      
    U, _, V = torch.svd(Omega)
    Omega = torch.mm(U, V.t())
    
    return Omega

def unitPerturb(std, matrixSize):
    Omega = gaussianElement(std, matrixSize)      
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())

    length = np.random.uniform(1-std*1e-3,1+std*1e-3, w.shape[0])
    angle = np.pi * np.random.uniform(0, 2, w.shape[0])

    x = length * np.cos(angle)
    y = length * np.sin(angle)

    w.real = x
    w = w + np.zeros(w.shape[0], dtype=complex)
    w.imag = y
    
    return torch.from_numpy(reconstruct_operator(w,v).real).float()

def unitary(std, matrixSize):
    
    return 'unitary'

# def gaussianElement(std, matrixSize):
#     gaussianMatrix = gaussianElement(std, matrixSize)
#     w, v = np.linalg.eig(Omega.cpu().detach().numpy())

#     w.real = np.random.uniform(-maxmin,maxmin, w.shape[0])
#     imag_dist = np.random.uniform(-maxmin,maxmin, w.shape[0])
#     w = w + np.zeros(w.shape[0], dtype=complex)
#     w.imag = imag_dist
    
#     return Omega