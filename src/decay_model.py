import os
from pyexpat import features
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse, r2_score

from src.config import Ns, betas, halflife_vals, regularization_vals
from src.utils import BETA_STR
# from src.surprise_signal_simulation import load_simulation

def simulate_auc(S, sigma):
    auc = S + np.random.normal(scale=sigma, size=len(S))
    if isinstance(auc, pd.Series):
        return auc.rename("AUC")
    return auc

def get_regression_features(df_model, ma_halflife):
    ma_t_minus_one = df_model["block"].ewm(
        halflife=ma_halflife, adjust=False).mean().shift(1).dropna() # shift caused dropping a value
    out_cols = {}
    out_cols['x1'] = df_model["block"] * ma_t_minus_one
    out_cols['x2'] = (1-df_model["block"]) * ma_t_minus_one
    out_cols['x3'] = df_model["block"]
    features = pd.DataFrame.from_dict(out_cols).dropna()
    return features

# def split_block(data, **split_kwargs):
#     train, test = train_test_split(data, **split_kwargs)
#     train = train.sort_index()
#     test = test.sort_index()
#     return train, test

# def fit_ridge(X, y, lambda_regularization):
#     reg = Ridge(alpha=lambda_regularization, random_state=123)
#     reg.fit(X, y)
#     return reg

def fit_decay_to_signal(df_model, auc_noise):
    best_mse = np.inf
    best_halflife = np.nan
    best_lambda = np.nan
    y = simulate_auc(df_model["S"], sigma=auc_noise).iloc[1:]   # we need t-1 for making explained variables
    for h in tqdm(halflife_vals, desc="halflife", leave=False):
        features = get_regression_features(df_model, ma_halflife=h)
        param_grid = {'alpha':regularization_vals}
        cv_search = GridSearchCV(Ridge(random_state=123), param_grid, cv=5, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=False)
        cv_search.fit(features, y)
        best_lambda_mse = cv_search.cv_results_['mean_test_score'][cv_search.best_index_] * (-1)    # sklearn implementation of scoring
        lambda_candidate = cv_search.best_params_["alpha"]
        if best_lambda_mse < best_mse:
            best_mse = best_lambda_mse
            best_halflife = h
            best_lambda = lambda_candidate
    return best_halflife, best_lambda

def convert_IB_to_decay(model_surprise_dict):
    # TODO: allow saving to file
    d_results = dict()
    for n in tqdm(Ns, desc = 'N'):
        d_results[n] = {}
        for beta_ind in tqdm(range(len(betas)), desc=BETA_STR, leave=False):
            df_model = model_surprise_dict[n, beta_ind]
            d_results[n][beta_ind] = fit_decay_to_signal(df_model.dropna(), auc_noise=0.0)
    
    return d_results

# if __name__=="__main__":
    
#     def extract_n_beta(filename):
#         name = os.path.splitext(os.path.basename(filename))[0]
#         N, beta_ind = int(name[2:4]), int(name[6:8])
#         return N, beta_ind

#     def load_simulation(path=os.path.join("data", "surprise_signal_from_simulated_block")):
#         fnames = [os.path.join(path, f) for f in os.listdir(path)]
#         model_surprise_dict = dict()
#         for filename in tqdm(fnames):
#             N, beta_ind = extract_n_beta(filename)
#             model_surprise_dict[(N, beta_ind)] = pd.read_parquet(filename)

#         return model_surprise_dict

#     model_surprise_dict = load_simulation()

#     N = 20
#     BETA_IND = 15
    
#     out = convert_IB_to_decay(model_surprise_dict)

            