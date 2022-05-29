# Koopman Autoencoders for cyclone prediction, conserved value discovery, and data synthesis
This repository is focused on three main tasks:
* Predicting the path of cyclones at a certain pressure level in the atmosphere
* Use a linear layer to approximate a Koopman operator (allows for linear stability analysis, eigenfunction discovery, etc.)
* Using Koopman eigenvectors to generate synthetic data for further improvement on prediction accuracy

This work is based on the paper [Forecasting Sequential Data Using Consistent Koopman Autoencoders](http://proceedings.mlr.press/v119/azencot20a/azencot20a.pdf).

## Path prediction
This script uses the L2-MSE loss (to adjust for latitutde and longitude on different parts of the globe) to train a general path prediction model. We will include links to baselines/current SOTAs here. The script plots the loss for both training and evaluation sets, and logs each experiment to W&B.
```bash
python3 train_predictive_net.py --model predictionANN --num_epochs 5 --batch_size 256 --synthetic True
```

## Stability analysis, conserved values

## Generation of synthetic data
### Constructing synthetic data from perturbation
Taking given examples, passing them through a trained model, perturbing the eigenvectors in the hidden state matrix, and then decoding these hidden states to generate new examples. $\mu$ and $\sigma$ are the mean and standard deviation of the distribution the random perturbation is sampled from. `choose_eigenvectors` is a user-provided function to determine which eigenvectors to perturb (default is the largest eigenvectors).
```bash
python3 data_synthesis.py --mu 0 --sigma 0.1 --model koopmanAE --choose_eigenvectors np.max
```

If you instead want to generate synthetic data through a GAN (generative adversarial network) rather than eigenvalue perturbation, run the following script:
```bash
python3 gan_synthesis.py --num_epochs 20 --batch_size 256
```

### GAN synthetic data evaluation
First, we train a classification model on real and fake cyclone images, so that it can distinguish between the two.
```bash
python3 classification_evaluator.py
```

This next script then runs the classification network on the synthetic data generated above, and removes cyclone images which it deems to be unlikely. The threshold for removal is 80% (probability that it is a real image). We can choose whether to run it on the GAN generated images or the perturbation-generated images (default is `perturbation`).
```bash
python3 synthetic_evaluation.py --threshold 0.8 --data perturbation
```

### Using synthetic data in prediction
Training a predictive neural network on raw, organic data vs organic + synthetic data. Will output plotted losses and comparison, as well as model structure. Uses the same script as general path prediction, but we pass in `synthetic=True` to train an additional model (same architecture) with the synthetic data. 
```bash
python3 train_predictive_net.py --model predictionANN --num_epochs 5 --batch_size 256 --synthetic True
```


