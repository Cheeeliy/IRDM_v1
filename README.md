# IRDM_v1
The code for "Imputed-Reconstruction Diffusion Models with Negative Exponential Noise Schedule for Multivariate Time Series Anomaly Detection"

# Train
You can train the model

python main.py --device cuda:0 --dataset [dataset]

The datasets supporting the results of this article are SMD, SWaT, MSL, PSM, and SMAP public datasets, and these are indicated in the manuscript with its website. And they should be chosen and preprocessed by min-max normalization before training.
