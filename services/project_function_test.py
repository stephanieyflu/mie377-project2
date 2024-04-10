from services.strategies import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def project_function_test(periodReturns, periodFactRet, x0, params):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    Strategy = BSS_MVO()
    llambda_to = params[0]
    L_c = params[1]
    U_c = params[2]
    K_c = params[3]
    L = params[4]
    U = params[5]
    K = params[6]
    
    if llambda_to == 0:
        min_to = False
    else:
        min_to = True
    
    x = Strategy.execute_strategy(periodReturns, periodFactRet,
                                            L=L, U=U, K=K,
                                            min_to=min_to, llambda_to=llambda_to, x0=x0,
                                            card=True, L_c=L_c, U_c=U_c, K_c=K_c)

    # Strategy = Mean_Variance_TE()
    # k = params[0]
    # min_to = params[1]
    # x = Strategy.execute_strategy(periodReturns, periodFactRet, k=k, x0=x0, min_to=min_to)

    # Strategy = PCA_MVO()
    # L_c = params[0]
    # U_c = params[1]
    # K_c = params[2]
    # min_to = params[3]
    # llambda_to = params[4]
    # x = Strategy.execute_strategy(periodReturns, p=3, 
    #                                         min_to=min_to, llambda_to=llambda_to, x0=x0,
    #                                         card=True, L_c=L_c, U_c=U_c, K_c=K_c)

    return x


def project_function_test2(periodReturns, periodFactRet, x0):
    """
    :param periodReturns:
    :param periodFactRet:
    :param x0:
    :return: x (weight allocation as a vector)
    """
    Strategy = RP()

    T, n = periodReturns.shape

    # Check if we are in the calibration period
    if T == 60: # 12 months * 5 years 
        # We are in the calibration period

        ##### Determine the optimal parameters #####

        # Range of parameters to test:
        cs = [5]
        Ls = [0]
        Us = [5, 6, 7, 8, 9, 10]
        Ks = [2, 3, 4, 5, 6, 7]

        all_params = [cs, Ls, Us, Ks]

        c_best, L_best, U_best, K_best = find_params(Strategy, all_params, periodReturns, periodFactRet, T, x0)

        params = pd.DataFrame({'c': [c_best], 'L': [L_best], 'U': [U_best], 'K': [K_best]})
        print(params)

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
            print("params_aes.csv deleted")
        
        params.to_csv('params_aes.csv', index=False)
        print("best params saved in params_aes.csv")

        params = pd.read_csv('params_aes.csv')
        c = params.iloc[0, 0]
        L = params.iloc[0, 1]
        U = params.iloc[0, 2]
        K = params.iloc[0, 3]

        x = Strategy.execute_strategy(48, periodReturns, periodFactRet, c=c, L=L, U=U, K=K)

    else: # No longer in the calibration period
        params = pd.read_csv('params_aes.csv')
        c = params.iloc[0, 0]
        L = params.iloc[0, 1]
        U = params.iloc[0, 2]
        K = params.iloc[0, 3]

        x = Strategy.execute_strategy(48, periodReturns, periodFactRet, c=c, L=L, U=U, K=K)

    return x

def find_params(Strategy, all_params, periodReturns, periodFactRet, T, x0):
    """Iterates through ps and Ks to determine the set of parameters that result in the optimal 
    Sharpe ratio during the calibration period

    Args:
        periodReturns (pd.DataFrame): asset returns during the calibration period
        T (int): number of data points (i.e., observations) in periodReturns

    Returns:
        best: 
    """
    cs = all_params[0]
    Ls = all_params[1]
    Us = all_params[2]
    Ks = all_params[3]
    
    SRs = []
    all_NumObs = []
    all_c = []
    all_L = []
    all_U = []
    all_K = []

    for w in [24, 36, 48]:
        for c in cs:
            for L in Ls:
                for U in Us:
                    for K in Ks:
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
                            subperiodFactRet = periodFactRet.iloc[start_index:end_index]

                            if t > 0:
                                # print(t)
                                # print(end_index)
                                portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)

                            weights = Strategy.execute_strategy(w, subperiodReturns, subperiodFactRet, c=c, L=L, U=U, K=K)
                        
                        SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                        SRs.append(SR[0])
                        all_NumObs.append(w)
                        all_c.append(c)
                        all_L.append(L)
                        all_U.append(U)
                        all_K.append(K)

    df = pd.DataFrame({'NumObs': all_NumObs, 'c': all_c, 'L': all_L, 'U': all_U, 'K': all_K, 'SR': SRs})

    ##### Save the optimal parameters #####
    df_avg = df.groupby(['c', 'L', 'U', 'K'])['SR'].mean().reset_index()
    max_index = df_avg['SR'].idxmax()
    c_best = df_avg.at[max_index, 'c']
    L_best = df_avg.at[max_index, 'L']
    U_best = df_avg.at[max_index, 'U']
    K_best = df_avg.at[max_index, 'K']

    return c_best, L_best, U_best, K_best

def plot(df, all_p):
    """
    Plots Sharpe ratio with respect to p, K and NumObs during the calibration period
    """
    clrs = sns.color_palette('hls', n_colors=10)
    
    #### PLOT 1 #####
    fig, ax1 = plt.subplots()
    plt.title('Sharpe Ratio with respect to p and K for PCA-OLS-MVO-CC')
    plt.suptitle('NumObs = 24')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sharpe Ratio')

    df1 = df[df['Window Size'] == 24]
    i = 0
    for p in sorted(list(set(all_p))):
        sub_df = df1[df1['p'] == p]
        # display(sub_df)
        ax1.plot(sub_df['K'].values, sub_df['SR'].values, label=str(p), marker='.', color=clrs[i])
        i += 1

    ax1.legend(title='p', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    ##### PLOT 2 #####
    fig, ax2 = plt.subplots()
    plt.title('Sharpe Ratio with respect to p and K for PCA-OLS-MVO-CC')
    plt.suptitle('NumObs = 36')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Sharpe Ratio')

    df1 = df[df['Window Size'] == 36]
    i = 0
    for p in sorted(list(set(all_p))):
        sub_df = df1[df1['p'] == p]
        # display(sub_df)
        ax2.plot(sub_df['K'].values, sub_df['SR'].values, label=str(p), marker='.', color=clrs[i])
        i += 1

    ax2.legend(title='p', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    ##### PLOT 3 #####
    fig, ax3 = plt.subplots()
    plt.title('Sharpe Ratio with respect to p and K for PCA-OLS-MVO-CC')
    plt.suptitle('NumObs = 48')
    ax3.set_xlabel('K')
    ax3.set_ylabel('Sharpe Ratio')

    df1 = df[df['Window Size'] == 48]
    i = 0
    for p in sorted(list(set(all_p))):
        sub_df = df1[df1['p'] == p]
        # display(sub_df)
        ax3.plot(sub_df['K'].values, sub_df['SR'].values, label=str(p), marker='.', color=clrs[i])
        i += 1

    ax3.legend(title='p', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    df.to_csv('abc.csv')
