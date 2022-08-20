import torch

def gaussianElement(std, matrixSize):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.tensor([std/matrixSize]))
    Omega = sampler.sample(matrixSize, matrixSize)[..., 0]
    return Omega