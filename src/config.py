import numpy as np

# Space of Parameters

# IB
Ns = np.arange(1, 50+1)
betas = np.logspace(0, 2, num=20, base=10)

# Regression
halflife_vals = Ns
regularization_vals = np.linspace(0, 2, num=20)

# noises
# noises = np.linspace(0.01,.09, num=9).round(2).tolist()
noises = [0.0, 0.001, 0.01, 0.1, 0.5, 0.1, 1, 3, 5]

# subjects_data

subjects_order = [
    'AT',
    'CK',
    'ES',
    'EW',
    'GG',
    'HLA',
    'IG',
    'LV',
    'MH',
    'MN',
    'MZ',
    'RS',
    'SG',
    'SM',
    'SN',
    'YF',
    'ZD'
    ]