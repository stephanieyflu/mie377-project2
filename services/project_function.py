from services.strategies import *
import os

def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = RP()

    T, n = periodReturns.shape

    # Check if we are in the calibration period
    if T == 60: # 12 months * 5 years 
        # We are in the calibration period

        ##### Determine the optimal parameters #####

        # Range of parameters to test:
        cs = [5]
        ps = list(range(1, 11))

        all_params = [cs, ps]

        c_best, p_best = find_params(Strategy, all_params, periodReturns, T)

        params = pd.DataFrame({'c': [c_best], 'p': [p_best]})
        print(params)

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
            print("params_aes.csv deleted")
        
        params.to_csv('params_aes.csv', index=False)
        print("best params saved in params_aes.csv")

        params = pd.read_csv('params_aes.csv')
        c = params.iloc[0, 0]
        p = params.iloc[0, 1]

        x = Strategy.execute_strategy(48, periodReturns, c=c, p=p)

    else: # No longer in the calibration period
        params = pd.read_csv('params_aes.csv')
        c = params.iloc[0, 0]
        p = params.iloc[0, 1]

        x = Strategy.execute_strategy(48, periodReturns, c=c, p=p)

    return x

def find_params(Strategy, all_params, periodReturns, T):
    """Iterates through ps to determine the set of parameters that result in the optimal 
    Sharpe ratio during the calibration period

    Args:
        periodReturns (pd.DataFrame): asset returns during the calibration period
        T (int): number of data points (i.e., observations) in periodReturns

    Returns:
        best: 
    """
    cs = all_params[0]
    ps = all_params[1]
    
    SRs = []
    all_NumObs = []
    all_c = []
    all_p = []

    for w in [24, 36, 48]:
        for c in cs:
            for p in ps:
                # Preallocate space for the portfolio per period value and turnover
                portfReturns = pd.DataFrame({'Returns': np.zeros(T)}, index=periodReturns.index)

                rebalancingFreq = 6
                windowSize = w

                numPeriods = (T - windowSize) // rebalancingFreq

                for t in range(numPeriods+1):
                    # Subset the returns and factor returns corresponding to the current calibration period.
                    start_index = t * rebalancingFreq
                    end_index = t * rebalancingFreq + windowSize
                    
                    subperiodReturns = periodReturns.iloc[start_index:end_index]

                    if t > 0:
                        portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)

                    weights = Strategy.execute_strategy(w, subperiodReturns, c=c, p=p)

                SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                SRs.append(SR[0])
                all_NumObs.append(w)
                all_c.append(c)
                all_p.append(p)

    df = pd.DataFrame({'NumObs': all_NumObs, 'c': all_c, 'p': all_p, 'SR': SRs})

    ##### Save the optimal parameters #####
    df_avg = df.groupby(['c', 'p'])['SR'].mean().reset_index()
    max_index = df_avg['SR'].idxmax()
    c_best = df_avg.at[max_index, 'c']
    p_best = df_avg.at[max_index, 'p']

    return c_best, p_best