import initLibrary, lossLibrary

class ExperimentCollection:
    def __init__(self, lossList, initList, stdList, datasetName, name):
        self.lossList = lossList
        self.initList = initList
        self.stdList = stdList
        self.datasetName = datasetName
        self.name = name
        self.collectionResults = {}
    
    def run(self, epochs=50, numRuns=3, batchSize=64, **kwargs):        
        for eigenLoss in self.lossList:
            for eigenInit in self.initList:
                print(f"### {eigenLoss} and {eigenInit} ###")
                for run in range(0, numRuns):
                    experiment = Experiment(eigenLoss, eigenInit, std)
                    self.collectionResults[eigenLoss][eigenInit][std][run] = experiment.run(
                        epochs = epochs,
                        batchSize = batchSize,
                        datasetName = self.datasetName
                    )
    
    def saveResults(self):
        with open(f"~/kae-cyclones/results/run_data/{self.name}.json", 'w') as f:
            json.dump(self.collectionResults, f)

class Experiment:
    def __init__(self, eigenLoss, eigenInit, std, datasetName, **kwargs):
        self.eigenLoss = eigenLoss
        self.eigenInit = eigenInit
        self.std = std
        self.datasetName = datasetName
    
    def run(self, epochs, batchSize):
        loss_dict = train()
        return None

class InitScheme:
    def __init__(self, distributionName, spread, matrixSize):
        self.distributionName = distributionName
        self.spread = spread
        self.matrixSize = matrixSize
    
    def __call__(self):
        return getInitFunc(self.distributionName)(self.spread, self.matrixSize)

class LossScheme:
    def __init__(self, lossName, weight):
        self.lossName = lossName
        self.weight = weight
    
    def __call__(self):
        return getInitFunc(self.distributionName)(self.weight)

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

def getLossFunc(lossName):
    if lossName == 'inverse':
        return lossLibrary.inverse