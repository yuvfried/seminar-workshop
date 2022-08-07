
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from information_bottleneck import IB



class BaseModel():
    def __init__(self):
        self.is_fitted = False

    def fit(self, *args):
        raise NotImplementedError

    def surprise_predictor(self, sequence):
        if not self.is_fitted:
            raise ValueError("must call 'fit' before 'surprise_predictor'")
    
    def score(self, *args):
        raise NotImplementedError
        

# Surprise values for a given N (rows) BETA (columns) pair
def load_IB_df_surprise():
    if os.name == "posix":
        filename = "data/df_surprise_huji_compatible.pkl"
    else:
        filename = "data/df_surprise.pkl"
    return pd.read_pickle(filename)

class IBModel(BaseModel):
    df_surprise = load_IB_df_surprise()

    def __init__(self, N, beta) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.S = None
    
    @staticmethod
    def calculate_pXY(N):
        pXY0 = [N-n+1 for n in range(N+1)]
        pXY1 = [n+1 for n in range(N+1)]
        denominator = (N+1)*(N+2)
        pXY = np.column_stack([pXY0,pXY1])/denominator
        return pXY

    @staticmethod
    def init_pXhat_X(N):
        return np.eye(N+1)
    
    def count_past_oddballs(self, sequence, pad=True):
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
        if isinstance(sequence, np.ndarray):
            sequence = pd.Series(sequence)
        # count num of past oddballs in a given window N
        ns = sequence.rolling(window=self.N, closed="left").sum().squeeze().rename("n")
        if not pad:
            ns = ns.dropna()
        return ns.values
    
    def fit(self, pXY=None, init_pXhat_X=None):
        d_results = IB(
            pXY=self.calculate_pXY(self.N),
            beta=self.beta,
            p0Xhat_X=self.init_pXhat_X(self.N)
            )
        pXhat_X = d_results["p(Xhat|X)"]
        pY_Xhat = d_results["p(Y|Xhat)"]
        self.S = -np.log2(pY_Xhat) @ pXhat_X
        self.is_fitted = True

    def surprise_predictor(self, sequence):
        super().surprise_predictor(sequence)
        ns = self.count_past_oddballs(sequence, pad=False).astype(np.int32)
        pad = np.full(self.N, np.nan)
        # df_seq_with_count = pd.DataFrame({"sequence": sequence, "n": ns})
        # S = df_seq_with_count.dropna().apply(
        #     lambda row: self.df_surprise.loc[self.N, self.beta][
        #         int(row["sequence"]), int(row["n"])], axis=1).reindex(ns.index)
        surprise = [self.S[t,n] for t, n in zip(sequence[self.N:], ns)]
        surprise = np.concatenate((pad, surprise))
        return surprise

class DecayModel(BaseModel):

    def __init__(self, tau) -> None:
        super().__init__()
        self.tau = tau
        self.reg = None
    
    def explanatory_variables(self, sequence):
        if isinstance(sequence, np.ndarray):
            sequence = pd.Series(sequence.squeeze())
        MA_t_minus_one = sequence.ewm(
            halflife=self.tau, adjust=True).mean().shift(1).dropna().values # shift caused dropping a value
        sequence = sequence[1:].values
        x1 = sequence * MA_t_minus_one
        x2 = (1-sequence) * MA_t_minus_one
        x3 = sequence
        return np.column_stack((x1,x2,x3))
    
    @staticmethod
    def __ignore_na(arr1, arr2):
        max_null_vals = np.maximum(
            np.isnan(arr1).any(axis=1).sum(),
            np.isnan(arr2).sum()
            )
        return arr1[max_null_vals:], arr2[max_null_vals:]
        
    
    def fit(self, response, sequence, na_policy="ignore"):
        X = self.explanatory_variables(sequence)
        y = response[1:]    # can't calculate MA(t-1) for t=0
        if na_policy == "ignore":
            X, y = self.__ignore_na(X, y)
        self.reg = LinearRegression().fit(X=X, y=y)
        self.is_fitted = True
    
    def score(self, X, y, na_policy="ignore"):
        if na_policy == "ignore":
            X, y = self.__ignore_na(X, y)
        r2_score = self.reg.score(X, y)
        return r2_score

    def surprise_predictor(self, sequence):
        super().surprise_predictor(sequence)
        X = self.explanatory_variables(sequence)
        return self.reg.predict(X)



