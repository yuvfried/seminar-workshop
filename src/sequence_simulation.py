import numpy as np
import random

global_np_seed_generator = np.random.default_rng(seed=123)

def create_block(p_oddball, block_size=240, name="block"):
    """Generate a random oddball block with p_oddball for oddball sound 
    (1) and 1-p for a frequent sound (0).

    Args:
        p_oddball (float): probability of oddball sound
        block_size (int, optional): Size of block. Defaults to 240.
        name (str, optional): name to assign for output Series. Defaults to "block".
        seed (int, optional): Seed for random generation of block. Defaults to 123.

    Returns:
        np.ndarray: array represents the block
    """    
    # sequence of frequent/oddball sounds
    block = global_np_seed_generator.choice([0, 1], p=[1 - p_oddball, p_oddball], size=block_size).astype(int)
    # return pd.Series(block, name=name) depracated
    return block

def create_sequence(p_oddballs = np.arange(0.1, 0.6, step=0.1)):
    """Keep random order of blocks, but within block all trails from same Bernully dist.
    I.e. block with p_pddball 0.1, then p=0.4 then p=0.2 ...

    Args:
        p_oddballs (_type_, optional): _description_. Defaults to np.arange(0.1, 0.6, step=0.1).

    Returns:
        _type_: _description_
    """
    blocks = [create_block(p) for p in p_oddballs]
    random.shuffle(blocks)    # shuffle blocks order
    sequence = np.concatenate(blocks)
    return sequence