from argparse import ArgumentParser
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import halflife_vals, Ns, betas, noises
from models import IBModel
from models import DecayModel
from utils import BETA_STR
from sequence_simulation import create_sequence


def time_print(msg):
    return print(datetime.now().strftime("%H:%M:%S"), msg)

def fit_tau_to_sequence(sequence, response):
    sequence, response = [arr[Ns.max():] for arr in [sequence, response]] # drop 50 firsts as they come from IB surprise
    best_tau = -1
    best_r2 = -1
    for tau in tqdm(halflife_vals, desc="tau", leave=False):
        model = DecayModel(tau)
        model.fit(response, sequence)
        surprise = model.surprise_predictor(sequence)
        weights = model.empiric_weights(sequence, num_bins=30)
        try:
            r2_score = model.r2(response, surprise, weights)
        except np.linalg.LinAlgError:
            continue
        if r2_score > best_r2:
            best_r2 = r2_score
            best_tau = tau
    
    return best_tau, best_r2

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", default=None)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    if args.name is None:
        print("dry run")
        exp_path = os.path.join("data", "dry_exp")
        os.makedirs(exp_path, exist_ok=True)
        Ns = np.arange(1, 11)
        betas = np.arange(1,6)
    else:
        exp_path = os.path.join("data", args.name)
        if os.path.exists(exp_path) and not args.override:
            raise ValueError(f"{exp_path} exists")
        else:
            os.makedirs(exp_path)

    # sequence generation
    time_print("simulate oddball sequence")
    sequence = create_sequence()
    # save sequence
    pd.Series(sequence).to_csv(os.path.join(exp_path, 
    "generated_sequence.csv"), index=False)

    # IB surprise predicotrs
    time_print("IB predictors")
    ib_out_path = os.path.join(exp_path, "IB_surprise_predictors")
    os.makedirs(ib_out_path, exist_ok=True)
    for N in tqdm(Ns, desc="N"):
        beta_surprise_dict = dict()
        for beta_ind in tqdm(range(len(betas)), desc=BETA_STR, leave=False):
            model = IBModel(N, beta_ind)
            model.fit()
            surprise = model.surprise_predictor(sequence)
            beta_surprise_dict[beta_ind] = surprise.tolist()
        
        # save all surprise vals for specific N in one json
        with open(os.path.join(ib_out_path, f"N={N:02d}.json"), 'w') as jf:
            json.dump(beta_surprise_dict, jf)

    # N tau relationship
    time_print("N tau relationship")
    records = list()
    for N in tqdm(Ns, desc="N"):
        with open(os.path.join(ib_out_path, f"N={N:02d}.json"), 'r') as jf:
            beta_surprise_dict = json.load(jf)
        for beta_ind in tqdm(range(len(betas)), desc=BETA_STR, leave=False):
            IB_surprise = np.array(beta_surprise_dict[str(beta_ind)])
            for noise in tqdm(noises, desc="noise", leave=False):
                response = IB_surprise + rng.normal(scale=noise, size=len(IB_surprise))
                
                best_tau, best_r2 = fit_tau_to_sequence(sequence, response)
                records.append(
                    {"N": N, 
                    "beta_ind": beta_ind, 
                    "noise": noise,
                    "r2":best_r2, 
                    "tau":best_tau}
                )
    # save
    N_tau_relationship = pd.DataFrame.from_records(records)
    N_tau_relationship = N_tau_relationship.astype({col:np.int8 for col in ["N", "beta_ind", "tau"]})
    N_tau_relationship.to_parquet(
        os.path.join(exp_path, "N_tau_relationship"), partition_cols=["N", "beta_ind"])
    
