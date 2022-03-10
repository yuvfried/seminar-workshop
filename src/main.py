### A place for running scripts ###

import argparse
import config
from surprise_signal_simulation import load_simulation
from decay_model import convert_IB_to_decay
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--noise", 
        help="AUC noise for simulation", type=int, default=0.0)
    args = parser.parse_args()

    regularization_vals = config.regularization_vals
    model_surprise_dict = load_simulation()
    d_results = convert_IB_to_decay(model_surprise_dict, regularization_vals, auc_noise=args.noise)
    
    # save results as a df to disk
    if not os.path.exists("data"):
        os.mkdir("data")
    pd.DataFrame.from_dict(d_results, orient="index").to_csv(
        "data/convert_ib_to_decay", index=False)
