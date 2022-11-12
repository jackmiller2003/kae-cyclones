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
import numpy as np

def create_U(size):
    return np.random.randn(size[0], size[1])

def change_eigvalues(U, random_vars):
    λ, v =  np.linalg.eig(U)
    
    p = np.abs(λ)
    θ = np.angle(λ)
    new_p = random_vars
    
    m, m = U.shape

    for j in np.arange(m-1):
        if np.abs(λ[j] - np.conj(λ[j+1])) < 1e-15:
            new_p[j+1] = new_p[j]

    Ũ = v @ np.diag(new_p * np.exp(1j*θ)) @ np.linalg.inv(v)
    return Ũ

def unitPerturb(std, matrixSize):
    new_p = np.random.uniform(1 - std*1e-3, 1+std*1e-3, size=matrixSize)
    U = create_U((matrixSize,matrixSize))
    new_U = change_eigvalues(U, new_p)
    print(f"Imag: {np.average(new_U.imag)}")
    print(f"Real: {np.average(np.abs(new_U.real))}")
    print(new_U)
    return torch.from_numpy(new_U.real).float()

def uniformEigen(std, matrixSize):
    new_p = np.random.uniform(0, std, size=matrixSize)
    U = create_U((matrixSize,matrixSize))
    new_U = change_eigvalues(U, new_p)
    print(f"Imag: {np.average(new_U.imag)}")
    print(f"Real: {np.average(np.abs(new_U.real))}")
    print(new_U)
    return torch.from_numpy(new_U.real).float()

def spikeAndSlab(std, matrixSize):
    alpha = 0.7
    elements = [0,1]
    probabilities = [alpha, 1-alpha]
    new_ps = []
    c = np.random.choice(elements, matrixSize, p=probabilities)
    for ch in c:
        if ch == 1:
            r = np.random.uniform(1-std, 1, size=1)[0]
            print(f"r = {r}")
            new_ps.append(r)
        else:
            new_ps.append(1)
    new_p = np.array(new_ps)
    U = create_U((matrixSize,matrixSize))
    new_U = change_eigvalues(U, new_p)
    print(f"Imag: {np.average(new_U.imag)}")
    print(f"Real: {np.average(np.abs(new_U.real))}")
    print(new_U)
    return torch.from_numpy(new_U.real).float()

# def unitPerturb(std, matrixSize):
#     new_p = np.random.uniform(1 - std*1e-3, 1+std*1e-3, size=matrixSize)
#     U = create_U(matrixSize, matrixSize)
#     new_U = change_eigvalues(U, new_p)
#     return new_U

# def uniformEigen(std, matrixSize):
#     new_p = np.random.uniform(0.5-std, 0.5+std, size=matrixSize)
#     U = create_U(matrixSize, matrixSize)
#     new_U = change_eigvalues(U, new_p)
#     return new_U

def gaussianElement(std, matrixSize):
    print(f"Gaussian std: {std/matrixSize}")
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/matrixSize]))
    Omega = sampler.sample((matrixSize, matrixSize))[..., 0]
    return Omega

def xavierElement(std, matrixSize):
    print(f"Xavier bounds: {np.sqrt(6/(matrixSize+matrixSize))}")
    sampler = torch.distributions.Uniform(torch.Tensor([-np.sqrt(6/(matrixSize+matrixSize))]), torch.Tensor([np.sqrt(6/(matrixSize+matrixSize))]))
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

# def unitPerturb(std, matrixSize):
#     Omega = gaussianElement(std, matrixSize)      
#     w, v = np.linalg.eig(Omega.cpu().detach().numpy())
#     x = np.random.uniform(1-(std*1e-1),1+(std*1e-1), w.shape[0]) #length * np.cos(angle)
#     y = np.random.uniform(1-(std*1e-1),1+(std*1e-1), w.shape[0]) #length * np.sin(angle)
#     w.real = x
#     w = w + np.zeros(w.shape[0], dtype=complex)
#     w.imag = y
#     return torch.from_numpy(reconstruct_operator(w,v).real).float()

# def gaussianElement(std, matrixSize):
#     gaussianMatrix = gaussianElement(std, matrixSize)
#     w, v = np.linalg.eig(Omega.cpu().detach().numpy())

#     w.real = np.random.uniform(-maxmin,maxmin, w.shape[0])
#     imag_dist = np.random.uniform(-maxmin,maxmin, w.shape[0])
#     w = w + np.zeros(w.shape[0], dtype=complex)
#     w.imag = imag_dist
    
#     return Omega