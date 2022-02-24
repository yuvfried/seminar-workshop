import numpy as np

# Space of Parameters

# IB
Ns = np.arange(1, 50+1)
betas = np.logspace(0, 2, num=20, base=10)

# Regression
halflife_vals = Ns
regularization_vals = np.linspace(0, 10, num=20)