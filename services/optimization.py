import cvxpy as cp
import numpy as np

from services.estimators import *
from services.optimization import *


def MVO(mu, Q):
    """
    #--- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
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

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
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

def risk_parity(mu, Q, c):

    '''
    Equalizes risk contribution for each asset

    Inputs:
    c = scaling coefficient for ln term

    Returns:
    Optimal weights of portfolio 
    '''

    # Find the total number of assets
    n = len(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Define and solve using CVXPY
    x = cp.Variable(n)
    y = cp.Variable(n)

    obj = cp.Minimize(((1 / 2) * cp.quad_form(y, Q)) - c*cp.sum(cp.log(y)))
    constraints = [y >= lb] #set constraints

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False, solver = cp.SCS)

    # Find individual y values from optimization and calculate sum
    y_val = np.array(y.value)
    y_sum = np.sum(y_val)

    # Normalize weights
    x = y_val / y_sum 

    return x


def rp_new(mu, Q, llambda, c):

    '''
    Equalizes risk contribution for each asset, takes into account expected returns

    Inputs:
    c = scaling coefficient for ln term
    llambda = scaling coefficient for expected returns

    Returns:
    Optimal weights of portfolio 
    '''

    # Find the total number of assets
    n = len(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Define and solve using CVXPY
    x = cp.Variable(n)
    y = cp.Variable(n)

    obj = cp.Minimize(((1 / 2) * cp.quad_form(y, Q)) - (llambda*cp.matmul(mu.T, y)) - c*cp.sum(cp.log(y)))
    constraints = [y >= lb] #set constraints

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)

    y_val = np.array(y.value)
    y_sum = np.sum(y_val)

    x = y_val / y_sum
    for i in range(n):
        k = y_val[i] * y_sum

    return x


def MVO_card_minT(mu, Q, x0, L, U, K, llambda):

    '''
    Returns estimated market portfolio weights.

    Inputs:
        L: lower bound of buy-in threshold (number) 
        U: upper bound of buy-in threshold (number)
        K: limit of # of assets you want in porftolio (number)
        llambda: coefficient of turnover function
    
    Returns:
        x (np.ndarray): n x 1 vector of estimated asset weights for the market portfolio
    '''


    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using Gurobi
    x = cp.Variable(n)
    y = cp.Variable(n, boolean = True)
    z = cp.Variable(n)

    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q) + llambda*cp.sum(z)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb,
                       z >= lb,
                       (cp.sum(y) <= K),
                       (x >= L*y), 
                       (x <= U*y),
                       (x - x0) <= z,
                       (x - x0) >= -z])

    prob.solve(solver=cp.GUROBI)
    return x.value




