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

# def gaussianElement(std, matrixSize):
#     gaussianMatrix = gaussianElement(std, matrixSize)
#     w, v = np.linalg.eig(Omega.cpu().detach().numpy())

#     w.real = np.random.uniform(-maxmin,maxmin, w.shape[0])
#     imag_dist = np.random.uniform(-maxmin,maxmin, w.shape[0])
#     w = w + np.zeros(w.shape[0], dtype=complex)
#     w.imag = imag_dist
    
#     return Omega