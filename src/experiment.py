import initLibrary
from train import *

class ExperimentCollection:
    def __init__(self, datasetName, name):
        self.datasetName = datasetName
        self.name = name
        self.collectionResults = {}
        self.runRegime = {}
    
    def run(self, epochs=10, numRuns=1, batchSize=64, **kwargs):        
        for eigenLoss, eigenInits in self.runRegime.items():
            self.collectionResults[eigenLoss] = {}
            for eigenInit, stds in eigenInits.items():
                self.collectionResults[eigenLoss][eigenInit] = {}
                for std in stds:
                    runDicts = []
                    for run in range(0, numRuns):
                        print(f"### {eigenLoss} and {eigenInit} and {std} and {run}###")
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
    
    def run(self, epochs=50, batchSize=128):
        train_ds, val_ds, _, train_loader, val_loader, input_size, alpha, beta, lr = create_dataset(self.datasetName, batchSize)
        init_scheme = InitScheme(self.eigenInit, self.std, beta)
        model = koopmanAE(init_scheme, beta, alpha, input_size)
        loss_dict = train(model, 0, train_loader, val_loader, len(train_ds), len(val_ds), lr, self.eigenLoss, epochs)
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
    elif distributionName == 'unitPerturbEigen':
        return initLibrary.unitPerturb
    
if __name__ == "__main__":
    # exp = Experiment("inverse", "gaussianElement", std=1, datasetName="ocean")
    # exp.run()

    expCol = ExperimentCollection('ocean', 'testRun')
    expCol.loadRunRegime('/home/156/jm0124/kae-cyclones/src/testingRegime.json')
    print(expCol.runRegime)
    expCol.run()
    print(expCol.collectionResults)
    expCol.saveResults()