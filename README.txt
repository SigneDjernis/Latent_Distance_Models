Latent_distance_models
Bachelor project
Signe Djernis Olsen (s206759)

##### Description
This project explores the Latent Distance Model (LDM), utilizing Bayesian statistics and
Maximum a Posteriori estimation for graph network analysis. It assesses LDM performance across diverse datasets
such as dolphin social networks, Dublin inhabitants and Facebook messages. Evaluation includes
examining the effects of prior additions and dimensionality variations in the latent space.


##### How to run the code
In this folder you will finde the necessary code for computing the LDM.
All the dataset folders contains a numbered order of runing the code,
1. is the import file, here the dataset is imported
2. is the file that generates the testdata and the points
3. is the file that models the data with different parameters
4. is the file that compares the optimal models with the baselines

For the folder FB-messages dataset is the files 3. the ones that compute and compare all 4 models
For the folder synthetic dataset is there a different system since there are no import file

The code should be run it the numbered order and you will get the same resulats as in the project.
Remeber you can change the parameters to get different outputs.


##### Description of all files
## Files
# Functions.py
The first file is Functions.py which conatins the Loss function and the gradient function both
with and without a prior. Aswell as the complex baseline that predicts whether two vertices are connected.
This file is the fundation of the project.

# Folder: Gradient Check
This folder has different files that test the corretness of the gradient function with and without a prior.
This is test in 3 different ways.

# Folder: Dolphins dataset
The files import the dataset and test the dimension and alhpa, by generating heatplots.
Aswell as an comparison of all 4 models

# Folder: Dublin dataset
The files import the dataset and test the dimension and alhpa, by generating curve plots.
Aswell as an comparison of all 4 models

# Folder: FB-massages dataset
The files import the dataset and compare of all 4 models one time

# Folder: Karate dataset
The files import the dataset and compute the latent space and a dendrogam for the LDM with and without a prior.

# Folder: Synthetic dataset
These files compute the socail space of a simple synthetic dataset both with the LDM with and without a prior.

# Other optimisation method
This file contains the two other optimisation methods that was used before MAP