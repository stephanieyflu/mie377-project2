import numpy as np
import pandas as pd
import cvxpy as cp


def OLS(returns, factRet):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q


def PCA(returns, p=3):
    '''
    Returns mu and Q estimates based on PCA.

    Inputs:
        returns (pd.DataFrame): T x n matrix of asset returns
        p (int): number of PCs to extract

    Returns:
        mu (np.ndarray): n x 1 vector of expected asset returns
        Q (np.ndarray): n x n matrix of asset covariances
    '''
    [T, n] = returns.shape

    ### PCA ###

    I = np.ones([T, 1])
    r_bar = (1/T) * ((returns.values).T @ I)

    # Centre the returns
    R_bar = returns.values - I @ (r_bar.T)

    # Estimate the biased covariance matrix
    Q_biased = (1/T) * (R_bar.T @ R_bar)

    # Perform eigenvalue decomposition
    w, v = np.linalg.eig(Q_biased) # w = eigenvalues, v = eigenvectors

    # Construct matrix of PCs
    P = R_bar @ v

    # Choose top p PCs
    P1 = P[:, :p]
    factRet = pd.DataFrame(np.real(P1))

    ### OLS ###

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0), 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q


def BSS(returns, factRet, U, L, K):
    # Number of observations and factors
    [T, p] = factRet.shape
    [T, n] = returns.shape

    B = np.zeros((p+1, n))

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # For loop to iterate through each asset
    for i in range(n):

        # Regression coefficients
        Bi = cp.Variable(p+1)
        y = cp.Variable(p+1, boolean=True)    
        ri = returns.iloc[:,i].values 

        obj = cp.Minimize(cp.sum_squares(ri - X @ Bi))
        constraints = [U*y >= Bi, 
                       L*y <= Bi,
                       sum(y) <= K
                       ] 

        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False, solver=cp.GUROBI)

        B[:, i] = Bi.value

    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0), 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q