import cvxpy as cp
import numpy as np
from scipy.stats import chi2


def MVO(mu, Q, 
        min_to=False, llambda_to=1, x0=[],
        robust=False, NumObs=36, alpha=0.95, llambda=1, 
        card=False, L_c=0.3, U_c=1, K_c=10):
    """
    #----------------------------------------------------------------------
    Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    
    Returns portfolio weights based on MVO.

    Inputs:
        mu (np.ndarray): n x 1 vector of expected asset returns
        Q (np.ndarray): n x n matrix of asset covariances
        robust (bool): flag for selecting robust MVO
        T (int): number of observations
        alpha (float): alpha value for ellipsoidal robust MVO
        llambda (float): lambda value for ellipsoidal robust MVO
    
    Returns:
        x (np.ndarray): n x 1 vector of estimated asset weights for the market portfolio

    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # Constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)

    if not min_to and not robust and not card:
        objective = (1 / 2) * cp.quad_form(x, Q)
        constraints = [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb]
    
    elif min_to and not robust and card:
        y = cp.Variable(n, boolean=True)
        z = cp.Variable(n)

        objective = ((1 / 2) * cp.quad_form(x, Q)) + (llambda_to * cp.sum(z))
        constraints = [A @ x <= b, 
                       Aeq @ x == beq, 
                       x >= lb,
                       z >= lb,
                       x - x0 <= z,
                       x - x0 >= -z,
                       (cp.sum(y) <= K_c),
                       (x >= L_c*y), 
                       (x <= U_c*y)]

    elif min_to and not robust and not card:
        z = cp.Variable(n)
        objective = ((1 / 2) * cp.quad_form(x, Q)) + (llambda_to * cp.sum(z))
        constraints = [A @ x <= b, 
                       Aeq @ x == beq, 
                       x >= lb,
                       z >= lb,
                       x - x0 <= z,
                       x - x0 >= -z]
        
    elif not min_to and robust and not card:
        # Calculate theta and epsilon for ellipsoidal robust MVO
        theta = np.sqrt((1/NumObs) * np.multiply(np.diag(Q), np.eye(n)))
        epsilon = np.sqrt(chi2.ppf(alpha, n))
        
        objective = ((1 / 2) * cp.quad_form(x, Q)) + (llambda * A @ x) + (epsilon * cp.norm(theta @ x, p=2))
        constraints = [Aeq @ x == beq,
                       x >= lb]
    
    elif not min_to and not robust and card:
        y = cp.Variable(n, boolean=True)
        objective = (1 / 2) * cp.quad_form(x, Q)
        constraints = [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb,
                       (cp.sum(y) <= K_c),
                       (x >= L_c*y), 
                       (x <= U_c*y)]
    
    elif not min_to and robust and card:
        y = cp.Variable(n, boolean=True)
        # Calculate theta and epsilon for ellipsoidal robust MVO
        theta = np.sqrt((1/NumObs) * np.multiply(np.diag(Q), np.eye(n)))
        epsilon = np.sqrt(chi2.ppf(alpha, n))
        
        objective = ((1 / 2) * cp.quad_form(x, Q)) + (llambda * A @ x) + (epsilon * cp.norm(theta @ x, p=2))
        constraints = [Aeq @ x == beq,
                       x >= lb,
                       (cp.sum(y) <= K_c),
                       (x >= L_c*y), 
                       (x <= U_c*y)]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(verbose=False)

    return x.value


def market_cap(r_mkt, R):
    '''
    Returns estimated market portfolio weights.

    Inputs:
        r_mkt (np.ndarray): T x 1 vector of market returns
        R (np.ndarray): T x n matrix of asset returns
    
    Returns:
        x (np.ndarray): n x 1 vector of estimated asset weights for the market portfolio
    '''
    T, n = R.shape

    # Define and solve using CVXPY
    x = cp.Variable(n)

    # Constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Disallow short sales
    lb = np.zeros(n)

    # Define objective function
    error = cp.norm(r_mkt - (R @ x), p=2)
    objective = cp.Minimize(error)

    # Define constraints
    constraints = [Aeq @ x == beq,
                   x >= lb]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)

    return x.value


def MV_TE(x_mkt, mu, Q, k, x0=[], min_to=False):
    n = len(mu)

    x = cp.Variable(n)
    y = cp.Variable(n, boolean=True)

    lb = np.zeros(n)
    Aeq = np.ones([1, n])

    objective = cp.quad_form(x - x_mkt, Q)
    constraints = [mu.T @ x >= mu.T @ x_mkt,
                   Aeq @ x == 1,
                   Aeq @ y <= k,
                   x <= y,
                   x >= lb]
    
    if min_to:
        z = cp.Variable(n)
        objective = cp.quad_form(x - x_mkt, Q) + cp.sum(z)
        constraints = [mu.T @ x >= mu.T @ x_mkt,
                       Aeq @ x == 1,
                       Aeq @ y <= k,
                       x <= y,
                       x >= lb,
                       z >= lb,
                       x - x0 <= z,
                       x - x0 >= -z]

    prob = cp.Problem(cp.Minimize(objective),
                      constraints)
    prob.solve(verbose=False)
    
    return x.value