from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from config import Ns, betas, halflife_vals
from utils import BETA_STR

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

def fit_decay_to_signal(df_model, regularization_vals, auc_noise):
    best_mse = np.inf
    best_halflife = np.nan
    best_lambda = np.nan
    y = simulate_auc(df_model["S"], sigma=auc_noise).iloc[1:]   # we need t-1 for making explained variables
    for h in tqdm(halflife_vals, desc="halflife", leave=False):
        features = get_regression_features(df_model, ma_halflife=h)
        param_grid = {'alpha':regularization_vals}
        cv_search = GridSearchCV(Ridge(random_state=123, solver='svd'), param_grid, cv=5, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=False)
        cv_search.fit(features, y)
        best_lambda_mse = cv_search.cv_results_['mean_test_score'][cv_search.best_index_] * (-1)    # sklearn implementation of scoring
        lambda_candidate = cv_search.best_params_["alpha"]
        if best_lambda_mse < best_mse:
            best_mse = best_lambda_mse
            best_halflife = h
            best_lambda = lambda_candidate
    return best_halflife, best_lambda

def convert_IB_to_decay(model_surprise_dict, regularization_vals, auc_noise=0.0):
    # TODO: allow saving to file
    d_results = dict()
    for n in tqdm(Ns, desc = 'N'):
        d_results[n] = {}
        for beta_ind in tqdm(range(len(betas)), desc=BETA_STR, leave=False):
            df_model = model_surprise_dict[n, beta_ind]
            d_results[n][beta_ind] = fit_decay_to_signal(
                df_model.dropna(), regularization_vals, auc_noise)
    
    return d_results

            