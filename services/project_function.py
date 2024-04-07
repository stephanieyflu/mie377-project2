from services.strategies import *


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = BSS_MVO()
    U = 5
    L = 0
    K = 3
    x = Strategy.execute_strategy(periodReturns, periodFactRet, U, L, K, x0=x0, min_to=False)

    # Strategy = Mean_Variance_TE()
    # x = Strategy.execute_strategy(periodReturns, periodFactRet, k=10, x0, min_to=True)

    # Strategy = PCA_MVO()
    # x = Strategy.execute_strategy(periodReturns, x0, p=3, min_to=False)

    return x
