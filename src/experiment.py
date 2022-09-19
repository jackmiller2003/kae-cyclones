import initLibrary
from train import *

from models import *
import torch
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
                        loss_dict, model, test_steps, test_loader = experiment.run(epochs = epochs, batchSize = batchSize, return_model=True)
                        runDicts.append(loss_dict)
                        if run == 0: min_loss, min_model = np.min(loss_dict['fwd_val']), model
                        else:
                            if np.min(loss_dict['fwd_val']) <= min_loss:
                                min_model = model
                                min_loss = np.min(loss_dict['fwd_val'])
                    
                    
                    accuracy, test_std = test_accuracy(model, 0, test_loader, test_steps)
                    
                    torch.save(min_model.state_dict(), f'/g/data/x77/jm0124/models/koopman/iclr_paper_models_22/{self.datasetName}-{eigenLoss}-{eigenInit}-{round(min_loss*1e2,2)}')
                    
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
                    
                    finalDict['testAccuracy'] = accuracy.astype(float).tolist()
                    finalDict['testStd'] = test_std.astype(float).tolist()                        
                    self.collectionResults[eigenLoss][eigenInit][std] = finalDict
    
    def saveResults(self):
        with open(f"/home/156/jm0124/kae-cyclones/results/run_data/{self.name}.json", 'w') as f:
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
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, test_steps, input_size, alpha, beta, lr, eigenlossHyper = create_dataset(self.datasetName, batchSize)
        init_scheme = InitScheme(self.eigenInit, self.std, beta)
        print(f"Length: {int(len(train_ds[0][0]))}")

        model = koopmanAE(init_scheme, beta, alpha, input_size, spectral_norm=False, steps=int(len(train_ds[0][0])))

        loss_dict = train(model, 0, train_loader, val_loader, len(train_ds), len(val_ds), lr, self.eigenLoss, epochs, eigenlossHyper)
        if return_model: return loss_dict, model, test_steps, test_loader
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
    

""" if __name__ == "__main__":
    direct = os.getcwd()
    if direct[10:16] == 'jm0124': run_path = '/home/156/jm0124/kae-cyclones/src/testingRegime.json'
    else: run_path = '/home/156/cn1951/kae-cyclones/src/testingRegime.json'


    expCol = ExperimentCollection('pendulum', 'pendulumDiss9_2')
    expCol.loadRunRegime(run_path)
    print(expCol.runRegime)
    expCol.run(epochs=50, numRuns=5)
    print(expCol.collectionResults)
    expCol.saveResults()

if __name__ == "__main__":
    exp = Experiment("none", "gaussianEigen", 1.0, "pendulum")
    _, model, _, val_ds, _, val_loader = exp.run(epochs=100, return_model=True)

    # prediction

    # let's get what we're feeding in right first

    # first let's see what the model expects

    # see if the model can take one of these
    encoder_output = model.encoder(val_ds[0][0][0].float().to(0))
    print("Encoder output:")
    print(encoder_output)

    dynamics_output = model.dynamics(encoder_output)
    print("Dynamics output:")
    print(dynamics_output)

    decoder_output = model.decoder(dynamics_output).cpu().detach().numpy()[0][0]
    print("Decoder output:")
    print(decoder_output)

    target = val_ds[0][0][1].cpu().numpy()
    print("Target:")
    print(target)

    error = np.linalg.norm(decoder_output - target) / np.linalg.norm(target)
    print("Error: ")
    print(error)

    print("Length of validation DS batch:")
    print(len(val_ds[0][0]))

    print("Length of validation DS:")
    print(len(val_ds))

    predictions, errors = [val_ds[0][0][0]], []
    for i in range(len(val_ds[0][0])-1):
        encoder_output = model.encoder(predictions[i].float().to(0))
        dynamics_output = model.dynamics(encoder_output)
        decoder_output = model.decoder(dynamics_output).cpu().detach().numpy()[0][0]
        predictions.append(model.decoder(dynamics_output))
        target = val_ds[0][0][i+1].cpu().numpy()
        errors.append(np.linalg.norm(decoder_output - target) / np.linalg.norm(target))

    print("Errors:")
    print(errors)"""

if __name__ == "__main__":
    l = [
            #('ocean', 'trying_new_ocean_4')
            #('fluid', 'trying_new_fluid_7')
            ('cyclone-limited', 'trying_new_cyclone_4')
            # ('pendulum0-200', 'trying_new_200'),
            # ('pendulum0-100', 'trying_new_100')
            # ('pendulum0-30', 'trying_new_30_2')
        ]
    
    for (ds, saveName) in l:
        expCol = ExperimentCollection(ds, saveName)
        
        if ds.startswith('pendulum'):      
            expCol.loadRunRegime('/home/156/jm0124/kae-cyclones/src/testingRegimeInit.json')
            epochs = 75
        else:
            expCol.loadRunRegime('/home/156/jm0124/kae-cyclones/src/testingRegimeInit.json')
            epochs = 100
        print(expCol.runRegime)
        expCol.run(epochs=epochs, numRuns=3)
        print(expCol.collectionResults)
        expCol.saveResults()