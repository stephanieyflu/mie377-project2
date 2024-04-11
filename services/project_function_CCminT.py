from services.strategies import *

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def project_function(periodReturns, periodFactRet, x0):
    Strategy = MVO_CC_minT_new() # Use the CC w min turnover

    T, n = periodReturns.shape
    x_initial = x0
    # Check if we are in the calibration period
    if T == 60: # 12 months * 5 years 
        # We are in the calibration period

        ##### Determine the optimal parameters #####

        #llambda = 10
        #L_b = [0, 0.2, 0.4, 0.6, 0.8, 1]
        #L_b = [0, 0.2, 0.5, 1]
        L_b = [0.2]
        U_b = [2, 3, 5]
        K_b = range(1,5)
        Ks = 20
        #Us = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.8, 1]
        Us = [0.1, 0.15, 0.2, 0.25]
        llambdas = [0.1, 1, 100]

    
        U_best, llambdas_best, L_b_best, U_b_best, K_b_best, no_best = find_params(Us, Ks, llambdas, L_b, U_b, K_b, Strategy, periodReturns, periodFactRet, x_initial, T)

        params = pd.DataFrame({'U': [U_best], 'llabdas': [llambdas_best], 'L_b': [L_b_best], 'U_b':[U_b_best], 'K_b':[K_b_best], 'no': [no_best]})

        # print(params)
        print(params)

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
        
        params.to_csv('params_aes.csv', index=False)

        params = pd.read_csv('params_aes.csv')
        U_best = params.iloc[0, 0]
        llambdas_best = params.iloc[0, 1]
        L_b_best = params.iloc[0, 2]
        U_b_best = params.iloc[0, 3]
        K_b_best = params.iloc[0, 4]
        no_best = params.iloc[0, 5]

        x = Strategy.execute_strategy(periodReturns, periodFactRet, NumObs=no_best, x0=x0, L=0.05, U=U_best, K=20, llambda = llambdas_best, L_b=L_b_best, U_b=U_b_best, K_b=K_b_best)

    else: # No longer in the calibration period
        params = pd.read_csv('params_aes.csv') # Read the best parameters
        U_best = params.iloc[0, 0]
        llambdas_best = params.iloc[0, 1]
        L_b_best = params.iloc[0, 2]
        U_b_best = params.iloc[0, 3]
        K_b_best = params.iloc[0, 4]
        #K_best = params.iloc[0, 1]
        no_best = params.iloc[0, 5]

        x = Strategy.execute_strategy(periodReturns, periodFactRet, NumObs=no_best, x0=x0, L=0.05, U=U_best, K=20, llambda = llambdas_best, L_b=L_b_best, U_b=U_b_best,K_b=K_b_best)

    print(x)
    return x

def find_params(Us, Ks, llambdas, L_b, U_b, K_b, Strategy, periodReturns, periodFactRet, x_initial, T):
    """Iterates through ps and Ks to determine the set of parameters that result in the optimal 
    Sharpe ratio during the calibration period

    Args:
        ps (np.ndarray): range of p to test
        Ks (np.ndarray): range of K to test
        Strategy (Class): strategy used to calculate portfolio weights
        periodReturns (pd.DataFrame): asset returns during the calibration period
        T (int): number of data points (i.e., observations) in periodReturns

    Returns:
        (best_p, best_K, best_no): best p, K, and NumObs parameters based on 
                                    Sharpe ratio during the calibration period
    """
    SRs = []
    win_size = []
    all_p = []
    all_K = []
    all_L = []
    all_U = []
    all_llambda = []
    n = 1
    for w in [24, 36, 48]:
        for l_b in L_b:
            for u_b in U_b:
                for k_b in K_b:
                    for u in Us:
                        for l in llambdas:

                            print(n)
                            x0 = x_initial
                            # Preallocate space for the portfolio per period value and turnover
                            portfReturns = pd.DataFrame({'Returns': np.zeros(T)}, index=periodReturns.index)

                            rebalancingFreq = 6
                            windowSize = w # NumObs

                            numPeriods = (T - windowSize) // rebalancingFreq

                            for t in range(numPeriods+1):
                                # Subset the returns and factor returns corresponding to the current calibration period
                                start_index = t * rebalancingFreq
                                end_index = t * rebalancingFreq + windowSize
                                
                                subperiodReturns = periodReturns.iloc[start_index:end_index]
                                subFactorReturns = periodFactRet.iloc[start_index:end_index]

                                if t > 0:
                                    # Calculate the portfolio period returns
                                    portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)

                                weights = Strategy.execute_strategy(subperiodReturns, subFactorReturns, NumObs=w, x0=x0, L=0.05, U=u, K=Ks, llambda = l, L_b=l_b, U_b = u_b, K_b=k_b)
                                x0 = weights
                                #print(x0)
                            # Calculate and save the Sharpe ratio for the current combination of parameters
                            SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                            SRs.append(SR[0])
                            win_size.append(w)
                            all_p.append(u)
                            all_K.append(k_b)
                            all_L.append(l_b)
                            all_U.append(u_b)
                            all_llambda.append(l)
                            n = n + 1

    df = pd.DataFrame({'Window Size': win_size, 'U': all_p, 'Lambda': all_llambda, 'L_b': all_L, 'U_b': all_U, 'K_b': all_K, 'SR': SRs})
    #plot(df, all_p)

    df.to_csv('fff.csv')
    ##### Save the optimal parameters #####

    df_avg = df.groupby(['U', 'Lambda', 'L_b', 'U_b', 'K_b'])['SR'].mean().reset_index()
    # df_avg.to_csv('aaa.csv')
    max_index = df_avg['SR'].idxmax()
    best_U = df_avg.at[max_index, 'U']
    best_llambda = df_avg.at[max_index, 'Lambda']
    best_lb = df_avg.at[max_index, 'L_b']
    best_ub = df_avg.at[max_index, 'U_b']
    best_K = df_avg.at[max_index, 'K_b']
    best_no = 48

    return best_U, best_llambda, best_lb, best_ub, best_K, best_no

def plot(df, all_p):
    """
    Plots Sharpe ratio with respect to U, K and NumObs during the calibration period
    """
    clrs = sns.color_palette('hls', n_colors=10)
    
    #### PLOT 1 #####
    fig, ax1 = plt.subplots()
    plt.title('Sharpe Ratio with respect to U and K for MVO-CC-Min-Turnover')
    plt.suptitle('NumObs = 24')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sharpe Ratio')

    df1 = df[df['Window Size'] == 24]
    i = 0
    for u in sorted(list(set(all_p))):
        sub_df = df1[df1['U'] == u]
        # display(sub_df)
        ax1.plot(sub_df['K'].values, sub_df['SR'].values, label=str(u), marker='.', color=clrs[i])
        i += 1

    ax1.legend(title='U', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    ##### PLOT 2 #####
    fig, ax2 = plt.subplots()
    plt.title('Sharpe Ratio with respect to U and K for MVO-CC-Min-Turnover')
    plt.suptitle('NumObs = 36')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Sharpe Ratio')

    df1 = df[df['Window Size'] == 36]
    i = 0
    for u in sorted(list(set(all_p))):
        sub_df = df1[df1['U'] == u]
        # display(sub_df)
        ax2.plot(sub_df['K'].values, sub_df['SR'].values, label=str(u), marker='.', color=clrs[i])
        i += 1

    ax2.legend(title='U', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    ##### PLOT 3 #####
    fig, ax3 = plt.subplots()
    plt.title('Sharpe Ratio with respect to U and K for MVO-CC-Min-Turnover')
    plt.suptitle('NumObs = 48')
    ax3.set_xlabel('K')
    ax3.set_ylabel('Sharpe Ratio')

    df1 = df[df['Window Size'] == 48]
    i = 0
    for u in sorted(list(set(all_p))):
        sub_df = df1[df1['U'] == u]
        # display(sub_df)
        ax3.plot(sub_df['K'].values, sub_df['SR'].values, label=str(u), marker='.', color=clrs[i])
        i += 1

    ax3.legend(title='U', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    df.to_csv('abc.csv')
