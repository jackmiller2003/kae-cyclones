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
                        experiment = Experiment(eigenLoss, eigenInit, float(std), self.datasetName, otherModel=eigenLoss)
                        loss_dict, model, test_steps, test_loader = experiment.run(epochs = epochs, batchSize = batchSize, return_model=True)
                        runDicts.append(loss_dict)
                        if run == 0: min_loss, min_model = np.min(loss_dict['fwd_val']), model
                        else:
                            if np.min(loss_dict['fwd_val']) <= min_loss:
                                min_model = model
                                min_loss = np.min(loss_dict['fwd_val'])
                    
                    
                    accuracy, test_std = test_accuracy(min_model, 0, test_loader, test_steps)
                    
                    torch.save(min_model.state_dict(), f'/g/data/x77/cn1951/models/koopman/iclr_paper_models_22/{self.datasetName}-{eigenLoss}-{eigenInit}-{round(min_loss*1e2,2)}')
                    
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
    def __init__(self, eigenLoss:str, eigenInit:str, std, datasetName, otherModel, **kwargs):
        self.eigenLoss = eigenLoss
        self.eigenInit = eigenInit
        self.std = std
        self.datasetName = datasetName
        self.epochs = 50
        self.otherModel = otherModel
    
    def run(self, epochs=50, batchSize=128, return_model=False, return_time=False, wd=0.01):
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, test_steps, input_size, alpha, beta, lr, eigenlossHyper = create_dataset(self.datasetName, batchSize)
        init_scheme = InitScheme(self.eigenInit, self.std, beta)
        print(f"Length: {int(len(train_ds[0][0]))}")
        back = False

        if self.otherModel == 'AE':
            model = regularAE(init_scheme, beta, alpha, input_size, spectral_norm=False, steps=int(len(train_ds[0][0])))
        elif self.otherModel == 'FF':
            model = feedForward(init_scheme, beta, alpha, input_size, spectral_norm=False, steps=int(len(train_ds[0][0])))
        elif self.eigenLoss.endswith('consistent'):
            back = True

        model = koopmanAE(init_scheme, beta, alpha, input_size, spectral_norm=False, steps=int(len(train_ds[0][0])), back=back)

        loss_dict = train(model, int(0), train_loader, val_loader, len(train_ds), len(val_ds), lr, self.eigenLoss, epochs, eigenlossHyper)
        if return_model: return loss_dict, model, test_steps, test_loader
        if return_time: return loss_dict, epoch_times
        return loss_dict, eigvals

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
    elif distributionName == 'newInit':
        return initLibrary.newInit
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
    elif distributionName == 'spikeAndSlab':
        return initLibrary.spikeAndSlab
    elif distributionName == 'xavierElement':
        return initLibrary.xavierElement
    elif distributionName == 'kaimingElement':
        return initLibrary.kaimingElement
    
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

if __name__ == "__main__":
    # wd=0.01
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss_2, _ = ge_exp.run(epochs=20, wd=0.01)
    # wd=0.1
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss_3, _ = ge_exp.run(epochs=20, wd=0.1)
    # wd=1
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss_4, _ = ge_exp.run(epochs=20, wd=1.0)
    # our method
    eigen_exp = Experiment("origin_mse", "gaussianEigen", 1.0, "ocean")
    eigen_loss, _ = eigen_exp.run(epochs=20)
    
    
    colors = ['black', '#003DFD', '#b512b8', '#11a9ba', '#0d780f', '#f77f07']
    epochs = [x for x in range(len(ge_loss_2["fwd_val"]))]
    #plt.plot(epochs, ge_loss_1["fwd_val"], label=r"$\alpha=0.0$", color=colors[0])
    plt.plot(epochs, ge_loss_2["fwd_val"], label=r"$\alpha=0.01$", color=colors[1])
    plt.plot(epochs, ge_loss_3["fwd_val"], label=r"$\alpha=0.1$", color=colors[2])
    plt.plot(epochs, ge_loss_4["fwd_val"], label=r"$\alpha=1.0$", color=colors[3])
    plt.plot(epochs, eigen_loss["fwd_val"], label="Eigenloss (Origin MSE)", color=colors[4])
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.legend(loc="best")
    plt.savefig("weight_decay.pdf", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300) 
    

"""
if __name__ == "__main__":
    # GAUSSIAN ELEMENT
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss, ge_time = ge_exp.run(epochs=20, batchSize=124, return_time=True)
    # GAUSSIAN EIGEN
    eigen_exp = Experiment("origin_mse", "gaussianEigen", 1.0, "ocean")
    eigen_loss, eigen_time = eigen_exp.run(epochs=20, return_time=True)
    # UNIFORM EIGEN
    unif_exp = Experiment("origin_mse", "gaussianElement", 1.0, "ocean")
    unif_loss, unif_time = unif_exp.run(epochs=20, return_time=True)
    # UNIT PERTURB
    unit_exp = Experiment("none", "gaussianElement", 1.0, "ocean")
    unit_loss, unit_time = unit_exp.run(epochs=20, return_time=True)
    
    colors = ['black', '#003DFD', '#b512b8', '#11a9ba', '#0d780f', '#f77f07']
    plt.plot(ge_time, ge_loss["fwd_val"], label="No penalty", color=colors[0])
    plt.plot(unif_time, unif_loss["fwd_val"], label="Eigenloss only", color=colors[1])
    plt.plot(unit_time, unit_loss["fwd_val"], label="Eigeninit only", color=colors[2])
    plt.plot(eigen_time, eigen_loss["fwd_val"], label="Eigenloss and eigeninit", color=colors[3])
    plt.xlim(0, 30)
    plt.xlabel("Wall time (seconds)")
    plt.ylabel("Validation loss")
    plt.legend(loc="best")
    plt.savefig("test_time.pdf", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300) 
"""

"""    
if __name__ == "__main__":
    l = [
            ('ocean', 'trying_new_ocean_4')
            #('fluid', 'trying_new_fluid_7')
            #('cyclone-limited', 'trying_new_cyclone_4')
            # ('pendulum0-200', 'trying_new_200'),
            #('duffing-100', 'duffingTrying'),
            # ('fp-100', 'duffingTrying3')
            ('pendulum0-100', 'allInitPendulum')
            # ('pendulum0-30', 'trying_new_30_2')
        ]
    
    for (ds, saveName) in l:
        expCol = ExperimentCollection(ds, saveName)
        
        if ds.startswith('pendulum'):      
            expCol.loadRunRegime('/home/156/cn1951/kae-cyclones/src/testingRegimeInit.json')
            epochs = 75
        else:
            expCol.loadRunRegime('/home/156/cn1951/kae-cyclones/src/testingRegimeInit.json')
            epochs = 100
        print(expCol.runRegime)
        expCol.run(epochs=epochs, numRuns=3)
        print(expCol.collectionResults)
        expCol.saveResults()
"""
