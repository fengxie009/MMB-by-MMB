import numpy as np
from scipy.stats import norm
from math import log, sqrt

def CI_Test(X, Y, S, D, alpha):
    # Input arguments:
    # X, Y: the variables to check the conditional independence of
    # S: conditioning set
    # D: Data matrix (number_of_Samples * n)
    # alpha: the parameter of Fischer's Z transform

    # Output arguments:
    # CI: the conditional independence relation between X, Y,
    #    true if independent, false if dependent
    # p: p-value (The null hypothesis is for independence)
    X = int(X)
    Y = int(Y)
    if isinstance(S, np.ndarray):
        S = S.tolist()
    n = D.shape[0]
    DD = D[:, [X, Y, *S]]
    corr_matrix = np.corrcoef(DD, rowvar=False)
    try:
        inv = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
    r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
    if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r)

    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(n - len(S) - 3) * abs(Z)
    p_value = 2 * (1 - norm.cdf(abs(X)))
    CI = p_value > alpha

    return CI, p_value


