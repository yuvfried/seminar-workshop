
from itertools import count
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
from statsmodels.api import WLS, OLS, add_constant # TODO migrate
from information_bottleneck import IB



class BaseModel():
    def __init__(self):
        self.is_fitted = False

    def fit(self, *args):
        raise NotImplementedError

    def surprise_predictor(self, sequence):
        if not self.is_fitted:
            raise ValueError("must call 'fit' before 'surprise_predictor'")
    
    def empiric_weights(self, sequence, num_bins, pad=True):
        """Compute weights for surprise values by their counts in bins. 
        Larger bins result in lower weight (rare values affect more).

        Args:
            sequence (_type_): _description_
            num_bins (_type_): _description_
            pad (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        surprise = self.surprise_predictor(sequence)    
        surprise = self.skip_first_trials(surprise)
        counts, bins = np.histogram(surprise,bins=num_bins)
        ranks = np.digitize(surprise, bins=bins, right=False)
        ranks[ranks == num_bins+1] = num_bins
        ranks -= 1
        
        weights = 1/counts[ranks]
        weights /= weights.sum()
        if pad:
            weights = self.pad_first_trials(weights)
        return weights
    
    def r2(self, auc, surprise, weights=None):
        if weights is None:
            weights = 1.0
        else:
            weights = self.skip_first_trials(weights)
        
        auc, surprise = [self.skip_first_trials(arr) for arr in [auc, surprise]]
        
        return WLS(auc, surprise, weights=weights).fit().rsquared
    
    def weights(self):
        raise NotImplementedError
    
    def skip_first_trials(self, arr, num_trials):
        # can't calculate surprise predictor to some of first trials in both models
        return arr[num_trials:]
    
    def pad_first_trials(self, arr, num_trials):
        fill_value = None if arr.dtype == int else np.nan
        if len(arr.shape) > 1:
            pad = np.full(shape=(num_trials, arr.shape[1]), fill_value=fill_value)
            return np.vstack((pad, arr))
        else:
            pad = np.full(shape=(num_trials,), fill_value=fill_value)
            return np.concatenate((pad, arr))
        
class IBModel(BaseModel):

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
    
    def fit(self):
        d_results = IB(
            pXY=self.calculate_pXY(self.N),
            beta=self.beta,
            p0Xhat_X=self.init_pXhat_X(self.N)
            )
        pXhat_X = d_results["p(Xhat|X)"]
        pY_Xhat = d_results["p(Y|Xhat)"]
        self.S = -np.log2(pY_Xhat) @ pXhat_X
        self.is_fitted = True
        return self

    def surprise_predictor(self, sequence):
        super().surprise_predictor(sequence)
        ns = self.count_past_oddballs(sequence, pad=False).astype(np.int32)
        pad = np.full(self.N, np.nan)
        surprise = [self.S[t,n] for t, n in zip(sequence[self.N:], ns)]
        surprise = np.concatenate((pad, surprise))
        return surprise
    
    def skip_first_trials(self, arr):
        return super().skip_first_trials(arr, self.N)
    
    def pad_first_trials(self, arr):
        return super().pad_first_trials(arr, self.N)

    
    def inverse_probs(self, sequence):
        """
        From the Paper:
    "Since there was an unbalanced distribution over the surprise values (by definition,
    higher surprise values are rarer), we used a weighted linear regression with inverse-probability
    weighting"
        """
        def calc_inverse_probs_for_surprise(group):
            """
                From the Paper:
        "The inverse-probability 1/p(s) was calculated using the true asymptotic probabilities
        given by p(yt+1, xt) by summing over all probabilities with the same surprise value"
            """
            p = 0
            for trial in group["block"]:
                for n in group["n"]:
                    p+=self.S[trial,n]
            return 1/p
        
        surprise = self.surprise_predictor(sequence)
        n = self.count_past_oddballs(sequence)
        df_model = pd.DataFrame(dict(zip(["block", "n", "S"], [sequence.astype(int), n.astype(int), surprise]))).dropna()
        
        surprise_groups = df_model.groupby("S")
        weights_map = surprise_groups.apply(calc_inverse_probs_for_surprise)
        inverse_probs = df_model["S"].map(weights_map.to_dict()).values
        inverse_probs /= inverse_probs.sum()
        return self.pad_first_trials(inverse_probs)
        
        

class DecayModel(BaseModel):

    def __init__(self, tau) -> None:
        super().__init__()
        self.tau = tau
        self.reg = None
    
    def explanatory_variables(self, sequence, pad=True):
        if isinstance(sequence, np.ndarray):
            sequence = pd.Series(sequence.squeeze())
        MA_t_minus_one = sequence.ewm(
            halflife=self.tau, adjust=True).mean().shift(1).dropna().values # shift caused dropping a value
        sequence = sequence[1:].values
        x1 = sequence * MA_t_minus_one
        x2 = (1-sequence) * MA_t_minus_one
        x3 = sequence
        variables = np.column_stack((x1,x2,x3))
        if pad:
            variables = np.vstack((np.full(3, np.nan), variables))
        return variables
    
    # @staticmethod
    # def __ignore_na(arr1, arr2):
    #     max_null_vals = np.maximum(
    #         np.isnan(arr1).any(axis=1).sum(),
    #         np.isnan(arr2).sum()
    #         )
    #     return arr1[max_null_vals:], arr2[max_null_vals:]
    
    # @staticmethod
    # def skip_first_trial(arr):
    #     return arr[1:]  # trials are array's rows
        
    
    def fit(self, response, sequence, **fit_params):
        X = self.explanatory_variables(sequence)
        X, y = [self.skip_first_trials(arr) for arr in [X, response]]
        X = add_constant(X)
        self.reg = OLS(y, X).fit()
        self.is_fitted = True
        return self
    
    # def score(self, X, y, na_policy="ignore"):
    #     if na_policy == "ignore":
    #         X, y = self.__ignore_na(X, y)
    #     r2_score = self.reg.score(X, y)
    #     return r2_score

    def surprise_predictor(self, sequence, pad=True):
        super().surprise_predictor(sequence)
        X = self.explanatory_variables(sequence)
        X = add_constant(self.skip_first_trials(X))
        surprise = self.reg.predict(add_constant(X))
        if pad:
            surprise = self.pad_first_trials(surprise)
        return surprise

    def skip_first_trials(self, arr):
        return super().skip_first_trials(arr, 1)
    
    def pad_first_trials(self, arr):
        return super().pad_first_trials(arr, 1)

