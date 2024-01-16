# PMPG-PINN: An Alternative Basis for Physics-informed Learning of Incompressible Fluid Mechanics

## Introduction
The code in this repository features a Python implemention of proposed PMPG-PINN on the lid-driven cavity problem. The input data are spatial coordinates (x, y) only taken inside the domain boundaries.   

## Pre-requisites
The code was run successfully using Tensorflow>=2.6.0. In addition, scipy is necessary for implementing optimization algorithm

## Data
The dataset used for comparison is available in the "data" folder in order to ensure the reproducibility of the results. 
Please, get in touch using the email address for correspondance in the paper for any collaborations. 


## File Structure
 The file follow structure below:

    PMPG-LidDriven
    ├── pred
    ├── PINN_PMPG.py
    ├── lbfgs.py
    └── mainTrain.py

***mainTrain.py***: script for training PINNs model, 

***pred***: Folder to store the prediction results
 
***PINN_PMPG.py***: class of the PMPG- routine

***lbfgs.py***: optimizer based on L-BFGS algorithm

***train.py***: script for training PINNs model 

## Results: 
Running the mainTrain, will generate the velocity magnitude prediction.
