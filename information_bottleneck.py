from scipy.stats import entropy
import numpy as np

"""
Notation:
1. Single Distribution: 
    p(P1=p1) --> pP1
2. Joint Distribution
    p(P1=p1, P2=p2) --> pP1P2
3. Conditional Distribution
    p(P1|P2=p2) --> pP1_P2
"""


def kl_divergence(p1, p2, **kwargs):
    return entropy(p1, p2, **kwargs)


def cond_entropy(p1_p2, **kwargs):
    return np.sum(entropy(p1_p2, **kwargs))


def mutual_information(p1, p1_p2, **kwargs):
    return entropy(p1, **kwargs) - cond_entropy(p1_p2, **kwargs)


def mutual_information_for_loop(p1p2, p1, p2):
    out = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            if p1p2[i, j] == 0:
                continue
            out += p1p2[i,j] * np.log2(p1p2[i,j] / (p1[i]*p2[j]))
    return out


def update_compression_probabilities(pX, py_X, py_Xhat, beta):
    X_shape = pX.shape[0]
    Xhat_shape = py_Xhat.shape[1]
    out = np.empty(shape=[Xhat_shape, X_shape])
    for xhat in range(Xhat_shape):
        for x in range(X_shape):
            out[xhat, x] = pX[x] * np.exp(-beta*kl_divergence(py_X[:, x], py_Xhat[:, xhat], base=2))

    return out/out.sum(axis=0)

def lagrangian(mi_XhatX, mi_Xhaty, beta):
    return mi_XhatX - beta * mi_Xhaty

def IB_iteration(pXY, beta, p0Xhat_X):
    # 2.
    pX = np.sum(pXY, axis=1)  # marginal distribution of X
    # middle iterative equation
    pXhat = p0Xhat_X @ pX  # law of total probability: Σ(over x) [p(X=x)*P(Xhat|X=x)]

    # 3.
    # we need Y|X=X and X|Xhat=Xhat in order to calculate Y|Xhat=Xhat
    py_X = pXY.T / pX  # Y|X=x conditional distribution definition
    # apply Bayes rule for calculating X|Xhat
    pXhatX = p0Xhat_X * pX  # chain rule for calculating joint dist from conditional dist
    pX_Xhat = ((pXhatX).T / pXhat).T
    # last iterative equation
    pY_Xhat = py_X @ pX_Xhat

    # compute minimization
    py = np.sum(pXY, axis=0)  # for computing MI of y;Xhat
    pyXhat = pY_Xhat * pXhat
    mi_XhatX = mutual_information_for_loop(pXhatX, pXhat, pX)
    mi_Xhaty = mutual_information_for_loop(pyXhat, py, pXhat)

    # 1.
    # first iterative equation - update pXhat_X
    pXhat_X = update_compression_probabilities(pX, py_X, pY_Xhat, beta)

    return pXhat_X, pY_Xhat, mi_XhatX, mi_Xhaty


def IB(pXY, beta, p0Xhat_X, convergence_diff=1e-6, verbose=False, track_loss=False):
    pXhat_X = p0Xhat_X  # init compression probabilities
    loss_vals = []  # track loss values
    loss_diff = np.inf
    prev_loss = np.inf
    i = 0
    while loss_diff > convergence_diff:
        i += 1
        pXhat_X, pY_Xhat, mi_XhatX, mi_Xhaty = IB_iteration(pXY, beta, pXhat_X)
        loss = lagrangian(mi_XhatX, mi_Xhaty, beta)
        loss_diff = prev_loss - loss
        prev_loss = loss

        if track_loss:
            loss_vals.append(loss)
        if verbose:
            if (i+1) % 10 == 0:
                print(f"{i+1}: {loss}")



    return {
        "p(Xhat|X)": pXhat_X,
        "p(Y|Xhat)": pY_Xhat,
        "I(Xhat;X)": mi_XhatX,
        "I(Xhat;Y)": mi_Xhaty,
        "loss": np.array(loss_vals),
    }

