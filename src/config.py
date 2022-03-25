import numpy as np

# Space of Parameters

# IB
Ns = np.arange(1, 50+1)
betas = np.logspace(0, 2, num=20, base=10)

# Regression
halflife_vals = Ns
regularization_vals = np.linspace(0, 2, num=20)

# noises
noises = np.linspace(0,1, num=11).tolist() + [1.5,2,3,5]