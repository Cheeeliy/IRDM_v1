# IRDM_v1
The code for the paper named "IRDM"

# Train
You can train the model

python main.py --device cuda:0 --dataset [dataset]

The datasets supporting the results of this article are SMD, SWaT, MSL, PSM, and SMAP public datasets, and these are indicated in the manuscript with its website. And they should be chosen and preprocessed by min-max normalization before training.
