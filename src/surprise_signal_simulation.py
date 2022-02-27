import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from src.config import Ns, betas
from src.utils import BETA_STR

df_surprise = pd.read_pickle("data/df_surprise.pkl")

def create_block(p_oddball, block_size=240, name="block", seed=123):
    np.random.seed(seed)
    # sequence of frequent/oddball sounds
    block = np.random.choice([0, 1], p=[1 - p_oddball, p_oddball], size=block_size).astype(int)
    return pd.Series(block, name=name)


def count_past_oddballs(block, N, pad=True):
    if isinstance(block, np.ndarray):
        block = pd.Series(block)
    # count num of past oddballs in a given window N
    ns = block.rolling(window=N, closed="left").sum().astype(pd.Int32Dtype()).rename("n")
    if pad:
        return ns
    return ns.dropna()


def surprise_signal(block, N, beta_ind, ret_as_df=True):
    ns = count_past_oddballs(block, N)
    df_block_with_count = pd.DataFrame({"block": block, "n": ns})
    S = df_block_with_count.dropna().apply(
        lambda row: df_surprise.loc[N, beta_ind][row["block"], row["n"]], axis=1).reindex(ns.index)
    if ret_as_df:
        df_block_with_count["S"] = S
        return df_block_with_count
    return S

def simulate(path=os.path.join("data", "surprise_signal_from_simulated_block"), seed=0):
    # concatenate 5 blocks with different p_oddball
    p_oddballs = np.arange(0.1, 0.6, step=0.1)
    blocks = [create_block(p) for p in p_oddballs]
    block = np.concatenate(blocks)

    # calculate surprise signals for each model.
    for N in tqdm(Ns, desc="N"):
        for beta_ind in tqdm(np.arange(len(betas)), desc=BETA_STR, leave=False):
            df_model = surprise_signal(block, N, beta_ind, ret_as_df=True)
            df_model.to_parquet(os.path.join(path, f"n={N:02d}b={beta_ind:02d}.snappy.parquet"))


def extract_n_beta(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    N, beta_ind = int(name[2:4]), int(name[6:8])
    return N, beta_ind

def load_simulation(path=os.path.join("data", "surprise_signal_from_simulated_block")):
    fnames = [os.path.join(path, f) for f in os.listdir(path)]
    model_surprise_dict = dict()
    for filename in tqdm(fnames):
        N, beta_ind = extract_n_beta(filename)
        model_surprise_dict[(N, beta_ind)] = pd.read_parquet(filename)

    return model_surprise_dict
