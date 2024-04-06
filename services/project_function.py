from services.strategies import *


def project_function(periodReturns, periodFactRet):
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

    Strategy = Mean_Variance_TE()
    x = Strategy.execute_strategy(periodReturns, periodFactRet, k=20)
    return x
