### A place for running scripts ###

import argparse
import config
from surprise_signal_simulation import load_simulation
from decay_model import convert_IB_to_decay
from utils import time_print
import pandas as pd
import os

def experiment(noise):
    regularization_vals = config.regularization_vals
    model_surprise_dict = load_simulation()
    d_results = convert_IB_to_decay(
        model_surprise_dict, 
        regularization_vals, 
        auc_noise=noise)
    return d_results

def save_d_results(d_results, noise):    
    # save results as a df to disk
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/ib_to_decay"):
        os.mkdir("data/ib_to_decay")
    pd.DataFrame.from_dict(d_results, orient="index").to_csv(
        f"data/ib_to_decay/noise={noise}.csv", index=False)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--noise", 
    #     help="AUC noise for simulation", type=float, default=0.0)
    # args = parser.parse_args()

    for noise in config.noises:
        time_print(f"Calculating Regression for AUC noise of {noise}")
        d_results = experiment(noise=noise)
        save_d_results(d_results, noise)


