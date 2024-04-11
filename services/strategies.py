import numpy as np
from services.estimators import *
from services.optimization import *


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class

class RP:

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, NumObs, periodReturns, factorReturns, c, p, L, U, K):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * NumObs:, :]
        factRet = factorReturns.iloc[(-1) * NumObs:, :]

        if p != 0:
        # mu, Q = OLS(returns, factRet)
            mu, Q = PCA(returns, p=p)

        elif p == 0:
            mu, Q = BSS(returns, factRet, L=L, U=U, K=K)
        
        x = risk_parity(mu, Q, c)
        return x, Q

def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x

class BSS_MVO:
    """
    uses BSS to estimate the covariance matrix and expected return
    and MVO with cardinality constraints
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, 
                         NumObs=36, L=0, U=5, K=3,
                         min_to=False, llambda_to=1, x0=[],
                         card=False, L_c=0.3, U_c=1, K_c=10):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:

        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * NumObs:, :]
        factRet = factorReturns.iloc[(-1) * NumObs:, :]
        mu, Q = BSS(returns, factRet, L, U, K)

        x = MVO(mu, Q, 
                min_to=min_to, llambda_to=llambda_to, x0=x0, 
                card=card, L_c=L_c, U_c=U_c, K_c=K_c)
        return x
    

class Mean_Variance_TE:
    """_summary_
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, k=10, x0=[], min_to=False):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:

        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        
        mu, Q = OLS(returns, factRet)

        x_mkt = market_cap(factRet['Mkt_RF'].values, returns.values)

        x = MV_TE(x_mkt, mu, Q, k, x0=x0, min_to=min_to)
        return x
    

class PCA_MVO:
    """
    uses PCA to estimate the covariance matrix and expected return
    and MVO with cardinality constraints
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, 
                         NumObs=36, p=3, 
                         min_to=False, llambda_to=1, x0=[],
                         card=False, L_c=0.3, U_c=1, K_c=10):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param NumObs:
        :param p: number of PCs to select as factors
        :param robust:
        :param NumObs:
        :param alpha:
        :param llambda:
        :param card:
        :param L:
        :param U:
        :param K:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * NumObs:, :]
        mu, Q = PCA(returns, p=p)
        
        x = MVO(mu, Q, 
                min_to=min_to, llambda_to=llambda_to, x0=x0, 
                card=card, L_c=L_c, U_c=U_c, K_c=K_c)
        return x