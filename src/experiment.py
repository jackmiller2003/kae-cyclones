import initLibrary
from train import *

class ExperimentCollection:
    def __init__(self, datasetName, name):
        self.datasetName = datasetName
        self.name = name
        self.collectionResults = {}
        self.runRegime = {}
    
    def run(self, epochs=50, numRuns=3, batchSize=64, **kwargs):        
        for eigenLoss, eigenInits in self.runRegime.items():
            for eigenInit, stds in eigenInits.items():
                print(f"### {eigenLoss} and {eigenInit} ###")
                for std in stds:
                    for run in range(0, numRuns):
                        print(f"### {eigenLoss} and {eigenInit} and {std} and {run}###")
                        experiment = Experiment(eigenLoss, eigenInit, float(std), self.datasetName)
                        self.collectionResults[eigenLoss][eigenInit][std][run] = experiment.run(
                            epochs = epochs,
                            batchSize = batchSize
                        )
    
    def saveResults(self):
        with open(f"~/kae-cyclones/results/run_data/{self.name}.json", 'w') as f:
            json.dump(self.collectionResults, f)
    
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

    def plot(self):
        loss_dict = self.run()
        epochs = [x for x in range(len(loss_dict["loss"]))]
        plt.plot(epochs, loss_dict["fwd"], label="Forward")
        plt.show()

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
