import cvxpy as cp
import numpy as np


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


def CVaR(mu, Q, alpha):
    """
    Perform Conditional Value at Risk (CVaR) portfolio optimization.

    Args:
    - mu: Expected returns of assets
    - Q: Covariance matrix of asset returns
    - alpha: Confidence level (default value is 0.95)

    Returns:
    - x: Optimal portfolio weights
    """

    # Find the total number of assets
    n = len(mu)

    #Target return
    targetRet = np.mean(mu)
    
    # Number of historical scenarios
    S = Q.shape[0]

    # Define and solve using CVXPY
    x = cp.Variable(n)
    z = cp.Variable(S)
   
    # Define the objective function
    prob = cp.Problem(cp.Minimize(cp.sum(z)/((1 - alpha)*S)),
                      [z >= 0,
                       z >= -mu.T @ x ,
                       cp.sum(x) == 1,
                       -mu.T @ x >= -targetRet])

    prob.solve(verbose=True, solver=cp.ECOS)
    return x.value


import scipy.stats as stats

def CVaR2(mu, Q, alpha):
    """
    Perform Conditional Value at Risk (CVaR) portfolio optimization.

    Args:
    - mu: Expected returns of assets
    - Q: Covariance matrix of asset returns
    - alpha: Confidence level (default value is 0.95)

    Returns:
    - cvar: Conditional Value at Risk
    """

       # Find the total number of assets
    n = len(mu)

    #Target return
    targetRet = np.mean(mu)
    
    # Number of historical scenarios
    S = Q.shape[0]

    # Define and solve using CVXPY
    x = cp.Variable(n)
    z = cp.Variable(S)
   
    # Define the objective function
    prob = cp.Problem(cp.Minimize(cp.sum(z)/((1 - alpha)*S)),
                      [z >= 0,
                       z >= -mu.T @ x ,
                       cp.sum(x) == 1,
                       -mu.T @ x >= -targetRet])

    prob.solve(verbose=True, solver=cp.ECOS)
    return x.value



def DistbRobustCVaR(mu, Q, x0, alpha =0.95, radius=1):
    """
    Perform Distributionally Robust Conditional Value at Risk (CVaR) portfolio optimization.

    Args:
    - mu: Expected returns of assets
    - Q: Covariance matrix of asset returns
    - alpha: Confidence level (default value is 0.95)
    - radius: Radius of the Wasserstein ball

    Returns:
    - x: Optimal portfolio weights
    """

    # Find the total number of assets
    n = len(mu)

    # Find size of S
    S = Q.shape[0]

    targetRet = np.mean(mu)

    # Define and solve using CVXPY
    x = cp.Variable(n)
    z = cp.Variable(S)

    # Define the Problem and Constraints 
       # Define the objective function
    prob = cp.Problem(cp.Minimize(cp.sum(z)/((1 - alpha)*S)),
                      [z >= 0,
                       z >= -mu.T @ x ,
                       cp.sum(x) == 1,
                       -mu.T @ x >= -targetRet,
                       cp.norm(x - x0, p=2) <= radius])

    prob.solve(verbose=True, solver=cp.ECOS)
    return x.value



def MonteCarlo(mu, Q, num_samples=10000):
    # Ensure mu is a 1-dimensional array
    mu = np.squeeze(mu)

    # Generate random samples from multivariate normal distribution
    samples = np.random.multivariate_normal(mu, Q, size=num_samples)

    # Calculate portfolio returns for each sample
    portfolio_returns = samples.sum(axis=1)

    # Find the index of the sample with the highest return
    max_return_index = np.argmax(portfolio_returns)

    # Get the optimal portfolio weights corresponding to the sample with the highest return
    x = samples[max_return_index]

    # Normalize weights to sum up to 1
    x = x / np.sum(x)

    return x


def MonteCarlo_CVaR(mu, Q, alpha=0.95, num_samples=10000):
    # Flatten mu if it's a 2D array
    
    mu = np.squeeze(mu)

    # Generate random samples from multivariate normal distribution
    samples = np.random.multivariate_normal(mu, Q, size=num_samples)

    # Calculate portfolio returns for each sample
    portfolio_returns = samples.sum(axis=1)

    # Find the index of the sample with the highest return
    max_return_index = np.argmax(portfolio_returns)

    # Get the optimal portfolio weights corresponding to the sample with the highest return
    x = samples[max_return_index]

    targetRet = np.mean(mu)

    # Normalize weights to sum up to 1
    x = x / np.sum(x)
    n = len(mu)
    S = Q.shape[0]
    z = cp.Variable(n)

    # Define the objective function
    objective = cp.Minimize(cp.sum(z) / ((1 - alpha) * S))

    # Define constraints
    constraints = [
        z >= 0,
        z >= -cp.matmul(mu.T, x) ,
        cp.sum(x) == 1,
        -cp.matmul(mu.T, x) >= -targetRet
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    return x

def MonteCarlo_Distb_Rob_CVaR(mu, Q, x0, alpha=0.95, num_samples=10000, radius = 1):
    # Flatten mu if it's a 2D array
    
    mu = np.squeeze(mu)

    # Generate random samples from multivariate normal distribution
    samples = np.random.multivariate_normal(mu, Q, size=num_samples)

    # Calculate portfolio returns for each sample
    portfolio_returns = samples.sum(axis=1)

    # Find the index of the sample with the highest return
    max_return_index = np.argmax(portfolio_returns)

    # Get the optimal portfolio weights corresponding to the sample with the highest return
    x = samples[max_return_index]

    targetRet = np.mean(mu)

    # Normalize weights to sum up to 1
    x = x / np.sum(x)
    n = len(mu)
    S = Q.shape[0]
    z = cp.Variable(n)

    # Define the objective function
    objective = cp.Minimize(cp.sum(z) / ((1 - alpha) * S))

    # Define constraints
    constraints = [
        z >= 0,
        z >= -cp.matmul(mu.T, x) ,
        cp.sum(x) == 1,
        -cp.matmul(mu.T, x) >= -targetRet,
        cp.norm(x - x0, p=2) <= radius
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    return x

   

#did not consider  
   
def MinTurnMonteCarlo(mu, Q, num_samples=100, min_turnover=0.5):
    # Ensure mu is a 1-dimensional array
    mu = np.squeeze(mu)
    
    # Initialize variables to store optimal portfolio weights and minimum turnover
    min_turnover = float('inf')
    x = None

    # Generate random samples from multivariate normal distribution
    for _ in range(num_samples):
        # Generate random portfolio weights
        weights = np.random.dirichlet(np.ones(len(mu)))
        
        # Calculate turnover as the sum of absolute differences in portfolio weights
        turnover = np.sum(np.abs(weights - mu))
        
        # Check if turnover is less than the current minimum
        if turnover < min_turnover:
            min_turnover = turnover
            x = weights
    
    return x

def Max_Sharpe_Min_Turn(mu, Q, x0, llambda=1):
    #towards minimizing turnover
        # Find the total number of assets
    n = len(mu)

    # Define and solve using CVXPY
    y = cp.Variable(n)
    k = cp.Variable()
    z = cp.Variable(n)
     #scalar big just turn over, and small is sharpe 
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Q)+(llambda*cp.sum(z))), 
                      [np.transpose(mu)@y == 1,
                       np.transpose(np.ones(n))@y == k,
                       z >= y - (k*x0),
                       z >= (k*x0) -y,
                       k >= 0,
                       y >= 0])
    
    prob.solve(verbose=False, solver=cp.ECOS)
    x = y.value/k.value

    return x


def MVO2(mu, Q, x0, llambda):
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
    z = cp.Variable(n)

    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)+(llambda*cp.sum(z))),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb,
                       z >= x0 - x,
                       z>= x - x0])
    prob.solve(verbose=False)
    return x.value