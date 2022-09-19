import initLibrary
from train import *

from models import *
import os

from tqdm import tqdm

class ExperimentCollection:
    def __init__(self, datasetName, name):
        self.datasetName = datasetName
        self.name = name
        self.collectionResults = {}
        self.runRegime = {}
    
    def run(self, epochs=25, numRuns=2, batchSize=64, **kwargs):        
        for eigenLoss, eigenInits in self.runRegime.items():
            self.collectionResults[eigenLoss] = {}
            for eigenInit, stds in eigenInits.items():
                self.collectionResults[eigenLoss][eigenInit] = {}
                print(f"### {eigenLoss} and {eigenInit}")
                for std in stds:
                    runDicts = []
                    for run in tqdm(range(0, numRuns)):
                        experiment = Experiment(eigenLoss, eigenInit, float(std), self.datasetName)
                        runDicts.append(experiment.run(epochs = epochs, batchSize = batchSize))
                    
                    averagedDict = {}
                    finalDict = {}

                    for key in runDicts[0].keys():
                        averagedDict[key] = []

                    for dct in runDicts:
                        for key, lst in dct.items():
                            averagedDict[key].append(lst)
                        
                    for key, llst in averagedDict.items():
                        nlist = []
                        for i in range(0, len(llst[0])):
                            avg = 0
                            for l in llst:
                                avg += l[i]
                            
                            avg = avg/len(llst)
                            nlist.append(avg)

                        finalDict[key] = nlist

                    self.collectionResults[eigenLoss][eigenInit][std] = finalDict
    
    def saveResults(self):
        with open(f"/home/156/cn1951/kae-cyclones/results/run_data/{self.name}.json", 'w') as f:
            json.dump(self.collectionResults, f)

    def plotResults(self):
        with open(f"~/kae-cyclones/results/run_data/{self.name}.json", 'w') as f:
            results = json.load(f)
        fig, axs = plt.subplots(1,2,figsize=(15,15), dpi=300)
        epochs = len(results["average"]["gaussianElement"]["1e0"]["loss"])
        axs[0][0].plot(epochs, results["average"]["gaussianElement"]["1e0"]["fwd_val"], label="Gaussian Element")
        axs[0][1].plot(epochs, results["average"]["gaussianEigen"]["1e0"]["fwd_val"], label="Gaussian Eigen")
        for l in axs:
            for ax in l:
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Mean Squared Error')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.legend(loc=1)
        
    def loadRunRegime(self, regimeFileName):
        with open(regimeFileName, 'r') as f:
            self.runRegime = json.load(f)


class Experiment:
    def __init__(self, eigenLoss:str, eigenInit:str, std, datasetName, **kwargs):
        self.eigenLoss = eigenLoss
        self.eigenInit = eigenInit
        self.std = std
        self.datasetName = datasetName
        self.epochs = 50
    
    def run(self, epochs=50, batchSize=128, return_model=False):
        train_ds, val_ds, _, train_loader, val_loader, input_size, alpha, beta, lr, eigenlossAlpha = create_dataset(self.datasetName, batchSize)
        init_scheme = InitScheme(self.eigenInit, self.std, beta)

        if self.eigenLoss == 'spectralNorm':
            model = koopmanAE(init_scheme, beta, alpha, input_size, spectral_norm=True)
        else:
            model = koopmanAE(init_scheme, beta, alpha, input_size, spectral_norm=False)

        loss_dict = train(model, 0, train_loader, val_loader, len(train_ds), len(val_ds), lr, self.eigenLoss, epochs, eigenlossAlpha)
        if return_model: return loss_dict, model, train_ds, val_ds, train_loader, val_loader
        return loss_dict

    def __str__(self):
        return f"Experiment for {self.epochs} epochs. Eigenloss={self.eigenLoss}, eigeninit={self.eigenInit}"

class InitScheme:
    def __init__(self, distributionName, spread, matrixSize):
        self.distributionName = distributionName
        self.spread = spread
        self.matrixSize = matrixSize
    
    def __call__(self):
        return getInitFunc(self.distributionName)(self.spread, self.matrixSize)

def getInitFunc(distributionName):
    if distributionName == 'gaussianElement':
        return initLibrary.gaussianElement
    elif distributionName == 'gaussianEigen':
        return initLibrary.gaussianEigen
    elif distributionName == 'doubleGaussianEigen':
        return initLibrary.doubleGaussianEigen
    elif distributionName == 'uniformEigen':
        return initLibrary.uniformEigen
    elif distributionName == 'svdElement':
        return initLibrary.svdElement
    elif distributionName == 'unitPerturb':
        return initLibrary.unitPerturb
    
def prediction_errors(model, val_ds, pred_steps=100, starting=0):
    predictions, errors = [val_ds[starting][0][0]], []
    for i in range(pred_steps):
        encoder_output = model.encoder(predictions[i].float().to(0))
        dynamics_output = model.dynamics(encoder_output)
        decoder_output = model.decoder(dynamics_output).cpu().detach().numpy()[0][0]
        predictions.append(model.decoder(dynamics_output))
        target = val_ds[starting+i+1][0][0].cpu().numpy()
        errors.append(np.linalg.norm(decoder_output - target) / np.linalg.norm(target))
    return errors

def aggregate_prediction_errors(model, val_ds, pred_steps=100):
    error_array = []
    for j in range(30):
        error_array.append(prediction_errors(model, val_ds, pred_steps=pred_steps, starting=j))
    error_array = np.asarray(error_array)
    return error_array

def plot_aggregate_prediction_errors(name, labels, error_arrays):
    """Plots the prediction errors for a variety of different experiments (initial conditions)."""
    fig = plt.figure(figsize=(15,12))
    for (error_array, label) in zip(error_arrays, labels):
        plt.plot(error_array.mean(axis=0), 'o--', lw=3, label=label)
        #plt.fill_between(x=range(error_array.shape[1]),
                        #y1=np.quantile(error_array, .20, axis=0), 
                        #y2=np.quantile(error_array, .80, axis=0), alpha=0.2)

    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)

    plt.ylabel('Relative prediction error', fontsize=22)
    plt.xlabel('Time step', fontsize=22)
    plt.grid(False)
    plt.ylim([0.0,error_array.max()*2])
    fig.tight_layout()
    plt.legend(loc="best", prop={'size': 22})
    plt.savefig(name +'.png')
    plt.close()


def plot_errors(errors, name):
    """Plots a single experiment's prediction errors (unaggregated)."""
    error = np.asarray(errors)
    fig = plt.figure(figsize=(15,12))
    x = np.arange(1, len(errors)+1)
    plt.plot(x, error, 'o--', lw=3, label='', color='#377eb8')

    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)

    plt.ylabel('Relative prediction error', fontsize=22)
    plt.xlabel('Time step', fontsize=22)
    plt.grid(False)
    #plt.ylim(0.0,1.0)
    fig.tight_layout()
    plt.savefig(name +'.png', dpi=300)
    plt.show()

def run_prediction_errors():
    # GAUSSIAN ELEMENT
    ge_exp = Experiment("none", "gaussianElement", 1.0, "ocean")
    ge_model, ge_val_ds = ge_exp.run(epochs=10, return_model=True)
    ge_array = aggregate_prediction_errors(ge_model, ge_val_ds)
    # GAUSSIAN EIGEN
    eigen_exp = Experiment("none", "gaussianEigen", 1.0, "ocean")
    eigen_model, eigen_val_ds = eigen_exp.run(epochs=75, return_model=True)
    eigen_array = aggregate_prediction_errors(eigen_model, eigen_val_ds)

    # UNIFORM EIGEN
    unif_exp = Experiment("none", "uniformEigen", 1.0, "ocean")
    unif_model, unif_val_ds = unif_exp.run(epochs=75, return_model=True)
    unif_array = aggregate_prediction_errors(unif_model, unif_val_ds)

    # UNIT PERTURB
    unit_exp = Experiment("none", "unitPerturb", 1.0, "ocean")
    unit_model, unit_val_ds = unit_exp.run(epochs=75, return_model=True)
    unit_array = aggregate_prediction_errors(unit_model, unit_val_ds)


    # plot all prediction errors
    plot_aggregate_prediction_errors("aggregate_ocean_errors", 
                                     ["gaussianEigen", "uniformEigen", "unitPerturb", "gaussianElement"], 
                                     [eigen_array, unif_array, unit_array, ge_array])


#if __name__ == "__main__":
    #run_prediction_errors()

    
if __name__ == "__main__":
    l = [
            #('ocean', 'ocean_final'),
            ('cyclone-limited', 'cyclone_final'),
            #('fluid', 'fluid_final')  
            # ('pendulum0', 'pendulum0_overnight_noise_run_100')
            #('pendulum5', 'pendulum5_overnight_noise_run_100')
            # ('pendulum9', 'pendulum9_overnight_noise_run_16')
        ]
    
    for (ds, saveName) in l:
        expCol = ExperimentCollection(ds, saveName)
        
        if ds.startswith('pendulum'):      
            expCol.loadRunRegime('/home/156/cn1951/kae-cyclones/src/testingRegimeOvernight.json')
        else:
            expCol.loadRunRegime('/home/156/cn1951/kae-cyclones/src/testingRegimeOvernight.json')
        print(expCol.runRegime)
        expCol.run(epochs=200, numRuns=1)
        print(expCol.collectionResults)
        expCol.saveResults()
