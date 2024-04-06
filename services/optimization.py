import cvxpy as cp
import numpy as np

from services.estimators import *
from services.optimization import *


def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
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
    prob.solve(verbose=False, solver=cp.ECOS)

    y_val = np.array(y.value)
    y_sum = np.sum(y_val)

    #print(y_sum)
    x = y_val / y_sum
    for i in range(n):
        #print(y_val[i])
        k = y_val[i] * y_sum
        #print(k)

    return x

def grp(mu, Q, c, llambda):

    # Find the total number of assets
    n = len(mu)

    # Define and solve using CVXPY
    x = cp.Variable(n)
    z = cp.Variable()
    one = np.ones(n)

    e = np.eye(n)

    constraints = []

    for i in range(n):

        e_i = e[:,i]
        outer_prod = np.outer(e_i, e_i)
        R = 0.5*(outer_prod @ Q + Q @ outer_prod)

        constraints.append((1 + c)*z - cp.quad_form(x, R) >= 0)
        constraints.append(cp.quad_form(x,R) - (1-c)*z >= 0)

    constraints.append(one.T @ x == 1)


    obj = cp.Minimize((0.5* cp.quad_form(x, Q)) - llambda*cp.matmul(mu.T, x))

    #constraints = [(1 + c)*z - x.T*R*x >= 0, x.T*R*x - (1-c)*z >= 0, one.T*x == 1] #set constraints

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False, solver=cp.GUROBI)

    return x.value

def rp_new(mu, Q, llambda, c):

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

    #print(y_sum)
    x = y_val / y_sum
    for i in range(n):
        #print(y_val[i])
        k = y_val[i] * y_sum
        #print(k)

    return x


# def black_litterman(periodReturns, factorReturns, NumObs, mu, Q, llambda):

#     # Find market weights
#     T, n = periodReturns.shape

#     # get the last T observations
#     returns = periodReturns.iloc[(-1) * NumObs:, :]
#     factRet = factorReturns.iloc[(-1) * NumObs:, :]
#     x_mkt = market_cap(factRet['Mkt_RF'].values, returns.values)

#     llambda = mu*x_mkt - factRet['RF'] / x_mkt*Q*x_mkt

#     pi = llambda*Q*x_mkt

#     #Solve for mu_bar with views

#     #Use BL to solve for x






