from services.strategies import *


def project_function_test(periodReturns, periodFactRet, x0, params):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    # Strategy = BSS_MVO()
    # U = params[0]
    # L = params[1]
    # K = params[2]
    # min_to = params[3]
    # x = Strategy.execute_strategy(periodReturns, periodFactRet, U, L, K, x0=x0, min_to=min_to)

    Strategy = Mean_Variance_TE()
    k = params[0]
    min_to = params[1]
    x = Strategy.execute_strategy(periodReturns, periodFactRet, k=k, x0=x0, min_to=min_to)

    # Strategy = PCA_MVO()
    # x = Strategy.execute_strategy(periodReturns, x0, p=3, min_to=False)

    return x
