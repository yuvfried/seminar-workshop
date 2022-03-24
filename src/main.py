### A place for running scripts ###

import argparse
import config
from surprise_signal_simulation import load_simulation
from decay_model import convert_IB_to_decay
import pandas as pd
import os

def experiment(noise=0.0):
    regularization_vals = config.regularization_vals
    model_surprise_dict = load_simulation()
    d_results = convert_IB_to_decay(model_surprise_dict, regularization_vals, auc_noise=args.noise)
    
    # save results as a df to disk
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/ib_to_decay"):
        os.mkdir("data/ib_to_decay")
    pd.DataFrame.from_dict(d_results, orient="index").to_csv(
        f"data/ib_to_decay/noise={noise}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--noise", 
        help="AUC noise for simulation", type=float, default=0.0)
    args = parser.parse_args()

    for noise in [0.2,0.5,1.2, 1.5, 2, 3]:
        experiment(noise=noise)


