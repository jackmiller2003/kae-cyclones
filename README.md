# Koopman Autoencoders for cyclone prediction, conserved value discovery, and data synthesis
This repository is focused on three main tasks:
* Predicting the path of cyclones at a certain pressure level in the atmosphere
* Use a linear layer to approximate a Koopman operator (allows for linear stability analysis, eigenfunction discovery, etc.)
* Using Koopman eigenvectors to generate synthetic data for further improvement on prediction accuracy

This work is based on the paper [Forecasting Sequential Data Using Consistent Koopman Autoencoders](http://proceedings.mlr.press/v119/azencot20a/azencot20a.pdf).

## Path prediction

## Stability analysis, conserved values

## Generation of synthetic data
### Constructing synthetic data
Taking given examples, passing them through a trained model, perturbing the eigenvectors in the hidden state matrix, and then decoding these hidden states to generate new examples. $\mu$ and $\sigma$ are the mean and standard deviation of the distribution the random perturbation is sampled from. `choose_eigenvectors` is a user-provided function to determine which eigenvectors to perturb (default is the largest eigenvectors).
```bash
python data_synthesis.py --mu 0 --sigma 0.1 --model koopmanAE --choose_eigenvectors np.argmax
```

### Using synthetic data in prediction
Training a predictive neural network on raw, organic data vs organic + synthetic data. Will output plotted losses and comparison, as well as model structure. 
```bash
python train_predictive_net.py --model predictionANN --num_epochs 5 --batch_size 256
```


