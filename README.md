# Koopman Autoencoders for cyclone prediction, conserved value discovery, and data synthesis
This repository is focused on three main tasks:
* Predicting the path of cyclones at a certain pressure level in the atmosphere
* Use a linear layer to approximate a Koopman operator (allows for linear stability analysis, eigenfunction discovery, etc.)
* Using Koopman eigenvectors to generate synthetic data for further improvement on prediction accuracy

This work is based on the paper [Forecasting Sequential Data Using Consistent Koopman Autoencoders](http://proceedings.mlr.press/v119/azencot20a/azencot20a.pdf).

## Notes for Jacko
* 24/05/2022: I've restructured to SRC format (`input`, `src`, `results`). Most of the code is in `src`, and we'll keep cleaning it up as we go. Only thing I need to check is that all the path references still work (but you've been using Pathlib so that should be fine).
