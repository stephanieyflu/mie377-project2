from services.strategies import *


def project_function(periodReturns, periodFactRet):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    
    Strategy = HistoricalMeanVarianceOptimization()
    x = Strategy.execute_strategy(periodReturns, periodFactRet)
    
    #x = equal_weight(periodReturns)
    return x

