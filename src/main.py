### A place for running scripts ###

import config
from surprise_signal_simulation import load_simulation
from decay_model import convert_IB_to_decay
import pandas as pd

if __name__ == "__main__":
    regularization_vals = config.regularization_vals
    model_surprise_dict = load_simulation()
    d_results = convert_IB_to_decay(model_surprise_dict, regularization_vals, auc_noise=0.0)
    # save results as a df to disk
    pd.DataFrame.from_dict(d_results, orient="index").to_csv(
        "data/convert_ib_to_decay.csv", index=False)
    df = pd.read_csv("data/convert_ib_to_decay.csv")
