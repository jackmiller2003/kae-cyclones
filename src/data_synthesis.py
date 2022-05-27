import torch
import xarray
from datasets import *
from models import *
from dl_pipeline import *
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import copy


class DataSynthesis:

    def __init__(self, mu, sigma, model, choose_eigenvectors):
        self.mu = mu
        self.sigma = sigma 
        self.model = model
        self.choose_eigenvectors = choose_eigenvectors
        self.train_json_path = '/g/data/x77/ob2720/partition/train.json'
        self.valid_json_path = '/g/data/x77/ob2720/partition/valid.json'
        self.test_json_path = '/g/data/x77/ob2720/partition/test.json'

    def create_modified_model(self, perturb=True):
        """
        Perturbs the largest eigenvector of the hidden state matrix in a model.
        Replaces hidden state with perturbed Koopman matrix.
        """
        # get eigendecomposition of Koopman operator
        W = self.model.dynamics.dynamics.weight.cpu().detach().numpy()
        w, v = np.linalg.eig(W)
        min_i = self.choose_eigenvectors(w) # maximum eigenvalue
        model_modified = copy.deepcopy(self.model).to(0) # create copy of model
        
        # sample random vector (matches size of eigenvector)
        s = np.random.normal(self.mu, self.sigma, len(v[0]))
        if perturb:
            v[min_i] += s
        W_1 = self.reconstruct_operator(w,v).real
        
        # put the modified Koopman matrix back in the model
        model_modified.to(0)
        model_modified.dynamics.dynamics.weight = torch.nn.Parameter(torch.from_numpy(W_1))
        return model_modified.to(0)

    def generate_new_data(self):   
        """
        Enumerates over training dataset, applying modified model to generate new images.
        """
        # create the modified model
        modified_model = self.create_modified_model()

        # instantiate the datasets
        train_ds = CycloneDataset('/g/data/x77/ob2720/partition/train/', tracks_path=self.train_json_path,
                                  save_np=True, load_np=False)
        val_ds = CycloneDataset('/g/data/x77/ob2720/partition/valid/', tracks_path=self.valid_json_path, 
                                save_np=True, load_np=False, partition_name='valid')
        test_ds = CycloneDataset('/g/data/x77/ob2720/partition/test/', tracks_path=self.test_json_path,
                                 save_np=True, load_np=False, partition_name='test')

        # enumerate over inputs, create modified examples, and save
        for i, (cyclone_array, cyclone, j) in tqdm(enumerate(train_ds)):
            x_reg = model(torch.from_numpy(cyclone_array).to(0))[0][0].cpu().detach().numpy()
            np.save(f'/g/data/x77/cn1951/synthetic_datasets/base_synthesis/u/2/{sigma}/train/{cyclone}-{j+1}', x_reg)
            x_mod = modified_model(torch.from_numpy(cyclone_array).to(0))[0][0].cpu().detach().numpy()
            np.save(f'/g/data/x77/cn1951/synthetic_datasets/normal_perturb_synthesis/u/2/{sigma}/train/{cyclone}-{j+1}', x_mod)

    def get_eigendecomp(self, model):
        """
        Returns the eigenvalues and eigenvectors of the learned Koopman operator in the model.
        """
        tensor_array = model.dynamics.dynamics.weight
        koopman_operator = tensor_array.cpu().detach().numpy()
        w, v = np.linalg.eig(koopman_operator)
        return w, v

    def reconstruct_operator(self, w, v):
        """
        Recreate a matrix from its eigenvalues and eigenvectors.
        """
        R = np.linalg.inv(v)
        # create diagonal matrix from eigenvalues
        L = np.diag(w)
        # reconstruct the original matrix
        B = v.dot(L).dot(R)
        return B

if __name__ == "__main__":
    saved_models_path = '/home/156/cn1951/kae-cyclones/saved_models'
    model_dae = koopmanAE(32, steps=4, steps_back=4, alpha=8).to(0)
    model_dae.load_state_dict(torch.load(f'{saved_models_path}/dae-model-continued-0.4210312087978713.pt'))
    model_dae.to(0)
    def choose_eigenvectors(): return np.argmax
    data_synthesiser = DataSynthesis(mu=0, sigma=0.5, model_dae, choose_eigenvectors)
    data_synthesiser.generate_new_data()

