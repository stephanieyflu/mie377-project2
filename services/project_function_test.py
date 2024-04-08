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

    # Strategy = BSS_MVO()
    # U = params[0]
    # L = params[1]
    # K = params[2]
    # min_to = params[3]
    # x = Strategy.execute_strategy(periodReturns, periodFactRet, U=U, L=L, K=K, x0=x0, min_to=min_to)

    Strategy = Mean_Variance_TE()
    k = params[0]
    min_to = params[1]
    x = Strategy.execute_strategy(periodReturns, periodFactRet, k=k, x0=x0, min_to=min_to)

    # Strategy = PCA_MVO()
    # p = params[0]
    # min_to = params[1]
    # x = Strategy.execute_strategy(periodReturns, p=p, min_to=min_to, x0=x0)

    return x


def project_function_test2(periodReturns, periodFactRet, x0):
    """
    :param periodReturns:
    :param periodFactRet:
    :param x0:
    :return: x (weight allocation as a vector)
    """
    Strategy1 = PCA_MVO()
    Strategy2 = BSS_MVO()
    strategies = [Strategy1, Strategy2]

    T, n = periodReturns.shape

    # Check if we are in the calibration period
    if T == 60: # 12 months * 5 years 
        # We are in the calibration period

        ##### Determine the optimal parameters #####

        # Range of parameters to test:
        strats = [0, 1]
        L_cs = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
        U_cs = [0.1, 0.15, 0.2, 0.25, 0.3]
        K_cs = list(range(15, 21))
        llambdas = [0.01, 0.1, 1, 10, 100]
        nos = [24, 36, 48]
        ps = list(range(1, 11))
        Ls = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        Us = list(range(1, 6))
        Ks = list(range(1, 8))

        param_ranges = [strats, L_cs, U_cs, K_cs, llambdas, nos, ps, Ls, Us, Ks]

        best = find_params(param_ranges, strategies, periodReturns, periodFactRet, T, x0)


        params = pd.DataFrame({'p0': best[0],
                               'p1': best[1],
                               'p2': best[2],
                               'p3': best[3],
                               'p4': best[4],
                               'p5': best[5],
                               'p6': best[6],
                               'p7': best[7],
                               'p8': best[8],
                               'p9': best[9]})
            
        print(params)

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
            print("params_aes.csv deleted")
        
        params.to_csv('params_aes.csv', index=False)
        print("best params saved in params_aes.csv")

        params = pd.read_csv('params_aes.csv')
        p0 = params.iloc[0, 0]
        p1 = params.iloc[0, 1]
        p2 = params.iloc[0, 2]
        p3 = params.iloc[0, 3]
        p4 = params.iloc[0, 4]
        p5 = params.iloc[0, 5]
        p6 = params.iloc[0, 6]
        p7 = params.iloc[0, 7]
        p8 = params.iloc[0, 8]
        p9 = params.iloc[0, 9]

        Strategy = strategies[p0]

        if p0 == 0:
            x = Strategy.execute_strategy(periodReturns, NumObs=p5, p=p6, 
                                            min_to=True, llambda_to=p4, x0=x0,
                                            card=True, L_c=p1, U_c=p2, K_c=p3)
        elif p0 == 1:
            x = Strategy.execute_strategy(periodReturns, periodFactRet,
                                            NumObs=p5, L=p7, U=p8, K=p9,
                                            min_to=True, llambda_to=p4, x0=x0,
                                            card=True, L_c=p1, U_c=p2, K_c=p3)

    else: # No longer in the calibration period
        params = pd.read_csv('params_aes.csv')
        p0 = params.iloc[0, 0]
        p1 = params.iloc[0, 1]
        p2 = params.iloc[0, 2]
        p3 = params.iloc[0, 3]
        p4 = params.iloc[0, 4]
        p5 = params.iloc[0, 5]
        p6 = params.iloc[0, 6]
        p7 = params.iloc[0, 7]
        p8 = params.iloc[0, 8]
        p9 = params.iloc[0, 9]

        Strategy = strategies[p0]

        if p0 == 0:
            x = Strategy.execute_strategy(periodReturns, NumObs=p5, p=p6, 
                                            min_to=True, llambda_to=p4, x0=x0,
                                            card=True, L_c=p1, U_c=p2, K_c=p3)
        elif p0 == 1:
            x = Strategy.execute_strategy(periodReturns, periodFactRet,
                                            NumObs=p5, L=p7, U=p8, K=p9,
                                            min_to=True, llambda_to=p4, x0=x0,
                                            card=True, L_c=p1, U_c=p2, K_c=p3)

    return x

def find_params(params_ranges, strategies, periodReturns, periodFactRet, T, x0):
    """Iterates through ps and Ks to determine the set of parameters that result in the optimal 
    Sharpe ratio during the calibration period

    Args:
        periodReturns (pd.DataFrame): asset returns during the calibration period
        T (int): number of data points (i.e., observations) in periodReturns

    Returns:
        (best_p, best_K, best_no): best p, K, and NumObs parameters based on 
                                    Sharpe ratio during the calibration period
    """
    param0s = params_ranges[0]
    param1s = params_ranges[1]
    param2s = params_ranges[2]
    param3s = params_ranges[3]
    param4s = params_ranges[4]
    param5s = params_ranges[5]
    param6s = params_ranges[6]
    param7s = params_ranges[7]
    param8s = params_ranges[8]
    param9s = params_ranges[9]

    for i0 in param0s:
        Strategy = strategies[i0]
        
        SRs = []
        param0s_save = []
        param1s_save = []
        param2s_save = []
        param3s_save = []
        param4s_save = []
        param5s_save = []
        param6s_save = []
        param7s_save = []
        param8s_save = []
        param9s_save = []

        if i0 == 0: # PCA_MVO
            for i1 in param1s:
                for i2 in param2s:
                    for i3 in param3s:
                        for i4 in param4s:
                            for i5 in param5s:
                                for i6 in param6s:
                                    # Preallocate space for the portfolio per period value and turnover
                                    portfReturns = pd.DataFrame({'Returns': np.zeros(T)}, index=periodReturns.index)

                                    rebalancingFreq = 6
                                    windowSize = i5 # NumObs

                                    numPeriods = (T - windowSize) // rebalancingFreq

                                    for t in range(numPeriods+1):
                                        # Subset the returns and factor returns corresponding to the current calibration period
                                        start_index = t * rebalancingFreq
                                        end_index = t * rebalancingFreq + windowSize
                                        subperiodReturns = periodReturns.iloc[start_index:end_index]
                                        if t > 0:
                                            # Calculate the portfolio period returns
                                            portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)
                                        weights = Strategy.execute_strategy(subperiodReturns, NumObs=i5, p=i6, 
                                                                            min_to=True, llambda_to=i4, x0=x0,
                                                                            card=True, L_c=i1, U_c=i2, K_c=i3)
                                    
                                    # Calculate and save the Sharpe ratio for the current combination of parameters
                                    SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                                    SRs.append(SR[0])
                                    param0s_save.append(i0)
                                    param1s_save.append(i1)
                                    param2s_save.append(i2)
                                    param3s_save.append(i3)
                                    param4s_save.append(i4)
                                    param5s_save.append(i5)
                                    param6s_save.append(i6)
            
            # Save the parameters for each PCA_MVO Strategy iteration
            df_0 = pd.DataFrame({'param0': param0s_save,
                                 'param1': param1s_save,
                                 'param2': param2s_save,
                                 'param3': param3s_save,
                                 'param4': param4s_save,
                                 'param5': param5s_save,
                                 'param6': param6s_save})
            # plot(df, all_p)

        elif i0 == 1: # BSS_MVO
            for i1 in param1s:
                for i2 in param2s:
                    for i3 in param3s:
                        for i4 in param4s:
                            for i5 in param5s:
                                for i7 in param7s:
                                    for i8 in param8s:
                                        for i9 in param9s:
                                            # Preallocate space for the portfolio per period value and turnover
                                            portfReturns = pd.DataFrame({'Returns': np.zeros(T)}, index=periodReturns.index)

                                            rebalancingFreq = 6
                                            windowSize = i5 # NumObs

                                            numPeriods = (T - windowSize) // rebalancingFreq

                                            for t in range(numPeriods+1):
                                                # Subset the returns and factor returns corresponding to the current calibration period
                                                start_index = t * rebalancingFreq
                                                end_index = t * rebalancingFreq + windowSize
                                                subperiodReturns = periodReturns.iloc[start_index:end_index]
                                                subperiodFactRet = periodFactRet.iloc[start_index:end_index]
                                                if t > 0:
                                                    # Calculate the portfolio period returns
                                                    portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)
                                                weights = Strategy.execute_strategy(subperiodReturns, subperiodFactRet,
                                                                                    NumObs=i5, L=i7, U=i8, K=i9,
                                                                                    min_to=True, llambda_to=i4, x0=x0,
                                                                                    card=True, L_c=i1, U_c=i2, K_c=i3)
                                            
                                            # Calculate and save the Sharpe ratio for the current combination of parameters
                                            SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                                            SRs.append(SR[0])
                                            param0s_save.append(i0)
                                            param1s_save.append(i1)
                                            param2s_save.append(i2)
                                            param3s_save.append(i3)
                                            param4s_save.append(i4)
                                            param5s_save.append(i5)
                                            param7s_save.append(i7)
                                            param8s_save.append(i8)
                                            param9s_save.append(i9)
            
            # Save the parameters for each PCA_MVO Strategy iteration
            df_1 = pd.DataFrame({'param0': param0s_save,
                                 'param1': param1s_save,
                                 'param2': param2s_save,
                                 'param3': param3s_save,
                                 'param4': param4s_save,
                                 'param5': param5s_save,
                                 'param7': param7s_save,
                                 'param8': param8s_save,
                                 'param9': param9s_save})
            # plot(df, all_p)               

    ##### Determine and save the optimal parameters #####

    df_avg_0 = df_0.groupby(['param1', 'param2', 'param3', 'param4', 'param6'])['SR'].mean().reset_index()
    df_avg_0.to_csv('a0.csv')
    max_index = df_avg_0['SR'].idxmax()
    best0_sr = df_avg_0.at[max_index, 'SR']
    best0_param0 = 0
    best0_param1 = df_avg_0.at[max_index, 'param1']
    best0_param2 = df_avg_0.at[max_index, 'param2']
    best0_param3 = df_avg_0.at[max_index, 'param3']
    best0_param4 = df_avg_0.at[max_index, 'param4']
    best0_param5 = 36 # did 48 for Project 1
    best0_param6 = df_avg_0.at[max_index, 'param6']
    best0_param7 = None
    best0_param8 = None
    best0_param9 = None


    df_avg_1 = df_1.groupby(['param1', 'param2', 'param3', 'param4', 'param7', 'param8', 'param9'])['SR'].mean().reset_index()
    df_avg_1.to_csv('a1.csv')
    max_index = df_avg_1['SR'].idxmax()
    best1_sr = df_avg_1.at[max_index, 'SR']
    best1_param0 = 1
    best1_param1 = df_avg_1.at[max_index, 'param1']
    best1_param2 = df_avg_1.at[max_index, 'param2']
    best1_param3 = df_avg_1.at[max_index, 'param3']
    best1_param4 = df_avg_1.at[max_index, 'param4']
    best1_param5 = 36 # did 48 for Project 1
    best1_param6 = None
    best1_param7 = df_avg_1.at[max_index, 'param7']
    best1_param8 = df_avg_1.at[max_index, 'param8']
    best1_param9 = df_avg_1.at[max_index, 'param9']

    if best0_sr > best1_sr:
        best = [best0_param0, 
                best0_param1,
                best0_param2,
                best0_param3,
                best0_param4,
                best0_param5,
                best0_param6,
                best0_param7,
                best0_param8,
                best0_param9]
    
    else:
        best = [best1_param0, 
                best1_param1,
                best1_param2,
                best1_param3,
                best1_param4,
                best1_param5,
                best1_param6,
                best1_param7,
                best1_param8,
                best1_param9]

    return best

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
