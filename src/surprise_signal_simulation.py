import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from config import Ns, betas
from utils import BETA_STR
from information_bottleneck import IB

global_np_seed_generator = np.random.default_rng(seed=123)

# Surprise values for a given N (rows) BETA (columns) pair
def load_df_surprise():
    if os.name == "posix":
        filename = "data/df_surprise_huji_compatible.pkl"
    else:
        filename = "data/df_surprise.pkl"
    return pd.read_pickle(filename)


def create_block(p_oddball, block_size=240, name="block"):
    """Generate a random oddball block with p_oddball for oddball sound 
    (1) and 1-p for a frequent sound (0).

    Args:
        p_oddball (float): probability of oddball sound
        block_size (int, optional): Size of block. Defaults to 240.
        name (str, optional): name to assign for output Series. Defaults to "block".
        seed (int, optional): Seed for random generation of block. Defaults to 123.

    Returns:
        pd.Series: Series represents the block
    """    
    # sequence of frequent/oddball sounds
    block = global_np_seed_generator.choice([0, 1], p=[1 - p_oddball, p_oddball], size=block_size).astype(int)
    # return pd.Series(block, name=name) depracated
    return block


def count_past_oddballs(block, N, pad=True):
    """Counts the past oddball sounds for each timestamp in the block.

    Args:
        block (pd.Series or np.ndarray with shape (block_size,)): block for calculation.
        N (int): The ealiest timestamp for counting the oddballs. E.g N=3 accounts 
        for t-1, t-2 and t-3.
        pad (bool, optional): if true returning the block with its original size, with null 
        values padded in the first N-1 timestamps. Defaults to True.

    Returns:
        pd.Series: Series of oddballs counts for each timestamp in the block.
    """    
    if isinstance(block, np.ndarray):
        block = pd.Series(block)
    # count num of past oddballs in a given window N
    ns = block.rolling(window=N, closed="left").sum().astype(pd.Int32Dtype()).rename("n")
    if pad:
        return ns
    return ns.dropna()


def surprise_signal(block, N, beta_ind, ret_as_df=True):
    """Compute IB Surprise Value for each timestamp in the block.

    Args:
        block (pd.Series or np.ndarray with shape (block_size,)): block for calculation.
        N (int): N parameter of IB
        beta_ind (int): index of beta parameter of IB
        ret_as_df (bool, optional): if true returning a df with columns ["block", "n", "S"] 
        where "n" is the oddball counts until t-N and "S" is the corresponding surprise signal. If false, only returns "S" as a series.
        Defaults to True.

    Returns:
        pd.DataFrame if ret_as_df is True, pd.Series otherwish: Surprise signal.
    """    
    ns = count_past_oddballs(block, N)
    df_block_with_count = pd.DataFrame({"block": block, "n": ns})
    df_surprise = load_df_surprise()
    S = df_block_with_count.dropna().apply(
        lambda row: df_surprise.loc[N, beta_ind][row["block"], row["n"]], axis=1).reindex(ns.index)
    if ret_as_df:
        df_block_with_count["S"] = S
        return df_block_with_count
    return S

def simulate(path, noise=0.0):
    # mkdir if not exists
    if not os.path.exists(path):
        os.mkdir(path)
    
    # concatenate 5 blocks with different p_oddball
    p_oddballs = np.arange(0.1, 0.6, step=0.1)
    block = np.concatenate([create_block(p) for p in p_oddballs])
    global_np_seed_generator.shuffle(block)

    # calculate surprise signals for each model.
    for N in tqdm(Ns, desc="N"):
        for beta_ind in tqdm(np.arange(len(betas)), desc=BETA_STR, leave=False):
            df_model = surprise_signal(block, N, beta_ind, ret_as_df=True)
            # depracated see comment below in 'add_noise' function
            # df_model["S"] += add_noise(df_model["S"], sigma=noise)
            df_model.to_parquet(os.path.join(path, f"n={N:02d}b={beta_ind:02d}.snappy.parquet"))


def extract_n_beta(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    N, beta_ind = int(name[2:4]), int(name[6:8])
    return N, beta_ind

# Depraceated - noise should be outside this module, as it is the last step of the simulation and there is no other thing here that depend on it
# Currently is is in decay modek in function fit h to n...
# def add_noise(block: pd.Series, sigma: float = 0.0):
#     return block + global_np_seed_generator.normal(loc=sigma, size=len(block))

def load_simulation(path=os.path.join("data", "surprise_signal_from_simulated_block")):
    if os.listdir(path) == 0:
        raise ValueError(f"{path} is empty")
    fnames = [os.path.join(path, f) for f in os.listdir(path)]
    model_surprise_dict = dict()
    for filename in tqdm(fnames, desc="load_simulation"):
        N, beta_ind = extract_n_beta(filename)
        model_surprise_dict[(N, beta_ind)] = pd.read_parquet(filename)

    return model_surprise_dict
