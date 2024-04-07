from services.strategies import *


def project_function_test(periodReturns, periodFactRet, x0, params):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    Strategy = BSS_MVO()
    U = 5#params[0]
    L = 0#params[1]
    K = 3#params[2]
    min_to = True#params[3]
    llambda_to = params[0]
    x = Strategy.execute_strategy(periodReturns, periodFactRet, U, L, K, llambda_to=llambda_to, x0=x0, min_to=min_to)

    # Strategy = Mean_Variance_TE()
    # k = params[0]
    # min_to = params[1]
    # x = Strategy.execute_strategy(periodReturns, periodFactRet, k=k, x0=x0, min_to=min_to)

    # Strategy = PCA_MVO()
    # x = Strategy.execute_strategy(periodReturns, x0, p=3, min_to=False)

    return x
