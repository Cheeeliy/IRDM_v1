# IRDM_v1
This is the code for "Imputed-Reconstruction Diffusion Models with Negative Exponential Noise Schedule for Multivariate Time Series Anomaly Detection"
# Train
To reproduce the results mentioned in our paper, first, make sure you have torch and pyyaml installed in your environment. Then, use the following command to train:

python main.py --device cuda:0 --dataset [dataset]

The datasets supporting the results of this article are SMD, SWaT, MSL, PSM, and SMAP public datasets, and the datasets are indicated in the manuscript.
