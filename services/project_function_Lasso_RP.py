from services.strategies import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def project_function(periodReturns, periodFactRet, x0):
    Strategy = Lasso_RP() # Use Lasso with risk parity
    T, n = periodReturns.shape
    
    no_obs = 48
    c = 1

    # Check if we are in the calibration period
    if T == 60: # 12 months * 5 years 
        # We are in the calibration period
        ##### Determine the optimal parameters #####
        
        best_S = find_params(Strategy, periodReturns, periodFactRet, c, T = T)
        
        params = pd.DataFrame({'S': [best_S]})
        print(params)

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
        
        params.to_csv('params_aes.csv', index=False)

        params = pd.read_csv('params_aes.csv')
        S_best = params.iloc[0, 0]
        x = Strategy.execute_strategy(periodReturns, periodFactRet, no_obs, c, S_best)

        
    
    else: # No longer in the calibration period
        params = pd.read_csv('params_aes.csv') # Read the best parameters
        S_best = params.iloc[0, 0]

        x = Strategy.execute_strategy(periodReturns, periodFactRet, no_obs, c, S_best)

    return x



def find_params(Strategy, periodReturns, periodFactRet, c, T): #(Strategy, periodReturns, periodFactRet, c, T)
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
    Ss = []
    win_size = []
    n = 1
    c = 1
    S = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2,3,4,5, 10, 20, 50, 100]

    for w in [24, 36, 48]:

        for s in range(len(S)):
        
            print(n)
            #x0 = x_initial
            # Preallocate space for the portfolio per period value and turnover
            portfReturns = pd.DataFrame({'Returns': np.zeros(T)}, index=periodReturns.index)

            rebalancingFreq = 6
            windowSize = w # NumObs

            numPeriods = (T - windowSize) // rebalancingFreq
            print(numPeriods)
            for t in range(numPeriods+1):
                # Subset the returns and factor returns corresponding to the current calibration period
                start_index = t * rebalancingFreq
                end_index = t * rebalancingFreq + windowSize
                                    
                subperiodReturns = periodReturns.iloc[start_index:end_index]
                subFactorReturns = periodFactRet.iloc[start_index:end_index]

                if t > 0:
                    # Calculate the portfolio period returns
                    portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)

                weights = Strategy.execute_strategy(subperiodReturns, subFactorReturns, NumObs=w, c=c, S = S[s])
                #weights = equal_weight(subperiodReturns)
                                    
                #weights = Strategy.execute_strategy(subperiodReturns, factorReturns = None)
                                    
                # Calculate and save the Sharpe ratio for the current combination of parameters
                SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                SRs.append(SR[0])
                Ss.append(S[s])
                win_size.append(w)

    df = pd.DataFrame({'Window Size': win_size, 'S':Ss, 'SR': SRs})

    df.to_csv('LASSO_RP_CCC.csv')

    ##### Save the optimal parameters #####

    df_avg = df.groupby(['S'])['SR'].mean().reset_index()
    max_index = df_avg['SR'].idxmax()
    best_S = df_avg.at[max_index, 'S']


    return best_S

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
