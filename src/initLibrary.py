import torch

def gaussianElement(std, matrixSize):
<<<<<<< HEAD
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/matrixSize]))
    Omega = sampler.sample((matrixSize, matrixSize))[..., 0]
    
    return Omega

def gaussianElement(std, matrixSize):
    gaussianMatrix = gaussianElement(std, matrixSize)
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())

    w.real = np.random.uniform(-maxmin,maxmin, w.shape[0])
    imag_dist = np.random.uniform(-maxmin,maxmin, w.shape[0])
    w = w + np.zeros(w.shape[0], dtype=complex)
    w.imag = imag_dist
    
    return Omega

def gaussianElement(std, matrixSize):
    gaussianMatrix = gaussianElement(std, matrixSize)
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())

    w.real = np.random.uniform(-maxmin,maxmin, w.shape[0])
    imag_dist = np.random.uniform(-maxmin,maxmin, w.shape[0])
    w = w + np.zeros(w.shape[0], dtype=complex)
    w.imag = imag_dist
    
=======
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.tensor([std/matrixSize]))
    Omega = sampler.sample((matrixSize, matrixSize))[..., 0]
>>>>>>> 31370088283d4686275eac24c6e8848aea7bcb86
    return Omega