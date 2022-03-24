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

def fit_decay_to_signal(df_model, halflife, regularization_vals, auc_noise):
    y = simulate_auc(df_model["S"], sigma=auc_noise).iloc[1:]   # we need t-1 for making explained variables
    features = get_regression_features(df_model, ma_halflife=halflife)  # the model regression variables
    param_grid = {'alpha':regularization_vals}  # search space of different lambda's
    
    # fir regression for each lambda
    cv_search = GridSearchCV(Ridge(random_state=123, solver='svd'), param_grid, cv=5, 
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=False)
    cv_search.fit(features, y)

    best_lambda = cv_search.best_params_["alpha"]
    return best_lambda

def convert_IB_to_decay(model_surprise_dict, regularization_vals, auc_noise=0.0):
    # TODO: allow saving to file
    n_to_halflife_map = pd.read_json("data/N_halflife_map.json", typ="series")
    d_results = dict()
    for n in tqdm(Ns, desc = 'N'):
        d_results[n] = {}
        for beta_ind in tqdm(range(len(betas)), desc=BETA_STR, leave=False):
            df_model = model_surprise_dict[n, beta_ind]
            h = n_to_halflife_map.loc[n].item()
            d_results[n][beta_ind] = fit_decay_to_signal(
                df_model.dropna(), 
                halflife=h, 
                regularization_vals=regularization_vals, 
                auc_noise= auc_noise)
    
    return d_results

            