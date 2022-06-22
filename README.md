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
This is the foundation of the paper. The process is as follows:
1. Generate synthetic data by perturbing eigenvectors, using a randomly sampled vector. Alternatively, generate synthetic data using a generative adversarial model (GAN).
2. Train a classification model to distinguish between real and fake cyclone images (will need to probably pre-generate and label some data with fake and real images).
3. Red-mark images which do not pass the classification model (i.e. are not "cyclone" enough).
4. Train a limited-capacity model using original data, full synthetic data and self-evaluated synthetic data, and compare performance between the three.
### Constructing synthetic data from perturbation
Taking given examples, passing them through a trained model, perturbing the eigenvectors in the hidden state matrix, and then decoding these hidden states to generate new examples. $\mu$ and $\sigma$ are the mean and standard deviation of the distribution the random perturbation is sampled from. `choose_eigenvectors` is a user-provided function to determine which eigenvectors to perturb (default is the largest eigenvectors).
```bash
python3 data_synthesis.py --mu 0 --sigma 0.1 --model koopmanAE --choose_eigenvectors np.max
```

If you instead want to generate synthetic data through a GAN (generative adversarial network) rather than eigenvalue perturbation, run the following script:
```bash
python3 gan_synthesis.py --num_epochs 20 --batch_size 256
```

### Synthetic data evaluation
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

# Other Information

## Running Scripts on NCI

To run these scripts on NCI, be apart of the hh5 group and then run:

```
module use /g/data/hh5/public/modules
module load conda/analysis3-22.01
module load pytorch/1.10.0
```

## Syncing W&B

Use the script when in src:

```
wandb sync --sync-all wandb 
```
# Paper To-Do

Tests to run:
1. Eigen regularisation:
  - Mean eigenvalue
  - Max eigenvalue
  - Inverse min eigenvalue
2. Eigen initialisation
  - Uniform distribution
  - Gaussian
  - Double hump Gaussian
3. Cyclones
  - 2x2 matrix init, no init etc.
4. Damped pendulum
  - 2x2 matrix init, no init etc.
5. Fashion MNIST
  - 2x2 matrix init, no init etc.

Plots for each experiment (obviously won't all be included in paper). Most of these will be for four combinations: just Koopman AE, eigeninit, eigenloss, and both together. Will also include PCA init for Fashion-MNIST.
1. Forward loss (could be either validation or otherwise). Just need to show the smoother loss convergence with eigeninit and eigenloss.
2. Eigenloss, for all methods (i.e. even no eigenloss penalty for backprop will still likely decrease eigenloss).
3. Spectral radius plot in Argand plane.
4. Table of final metrics for each task (i.e. MSE for all four techniques combinations).

## Questions
1. Are we sure that we don't need a formulation of the eigenvalue loss that enables the calculation of derivatives for back-propagation? Are we sure that Pytorch handles this automatically for us?

___
## References
* [Forecasting Sequential Data Using Consistent Koopman Autoencoders](http://proceedings.mlr.press/v119/azencot20a/azencot20a.pdf).
* [A Koopman Approach to Understanding Sequence Neural Models](https://arxiv.org/abs/2102.07824)
* [Introduction to Koopman operator theory of dynamical systems](https://www.mit.edu/~arbabi/research/KoopmanIntro.pdf)
* [Deep Learning with Eigenvalue Decay Regularizer](https://arxiv.org/pdf/1604.06985.pdf?fbclid=IwAR2iY9cu3d_yxF_Kg_tM2dJeLe2DHJbc8CagUFuvXJhlYEG4WDRZrUJRXxY)
* [Spectral Properties of Dynamical Systems, Model Reduction and Decompositions](https://link.springer.com/content/pdf/10.1007/s11071-005-2824-x.pdf)
* [Nonlinear dynamics and chaos](https://www.fulviofrisone.com/attachments/article/464/Strogatz,%20S.H.%20-%20Nonlinear%20dynamics%20and%20chaos.pdf)
* [Data-driven spectral analysis for coordinative structures in periodic human locomotion](https://www.nature.com/articles/s41598-019-53187-1.pdf)
* [PCA-Initialized Deep Neural Networks Applied To Document Image Analysis](https://arxiv.org/pdf/1702.00177.pdf)
