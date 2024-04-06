from services.strategies import *


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    # For BSS
    U = 5
    L = 0
    K = 3

    Strategy = BSS_MVO()
    x = Strategy.execute_strategy(periodReturns, periodFactRet, U, L, K, x0, min_to=True)

    # Strategy = Mean_Variance_TE()
    # x = Strategy.execute_strategy(periodReturns, periodFactRet, x0, k=10, min_to=True)

    # Strategy = PCA_MVO()
    # x = Strategy.execute_strategy(periodReturns, x0, p=3, min_to=False)

    return x
