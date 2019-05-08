#%% Initialize Global Settings
import platform
import pandas as pd
dataDir = ''
if platform.system() == 'Windows':
    dataDir = 'F:\\modified_data'
elif platform.system() == 'Linux':
    dataDir = '/home/joseph/Desktop/datastore'
else:
    exit()

pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", '{:,.3f}'.format)

import sys
import os.path as path
from forex.OHLC import OHLC

currencyPairs = ['AUDUSD', 'EURGBP', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'XAUUSD']
if len(sys.argv) > 1:
    currencyPairs = [sys.argv[1]]
print(currencyPairs)


for currencyPair in currencyPairs:
    p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
    p.set_df(p.get_df_with_resolution('1min'))
    pipReturn1Interval = p.df.Close.diff() * 10000
    p.df['PIP_Return'] = pipReturn1Interval.shift(-1)
    p.df.dropna(inplace=True)
    p.df.reset_index(drop=True, inplace=True)
    # p.df = p.df.iloc[:1000,:]

    tmp = p.df[['Close', 'PIP_Return']].copy()
    interval = 60

    # lossDrawWin = [] # -1 = Loss; 0 = Draw; 1 = Earn
    simulatedReturns = []

    buyHoldSell = [] # -1 = Sell; 0 = Hold; 1 = Buy

    earnThreshold = 400
    lossThreshold = -100
    if currencyPair in ['EURUSD', 'AUDUSD']:
        earnThreshold = 20
        lossThreshold = -5

    lastIndex = len(tmp.index)-interval-1
    for i in range(0, len(tmp.index)):
        if i > lastIndex:
            # lossDrawWin.append(0)
            buyHoldSell.append(0)
            simulatedReturns.append(0)
            continue

        if i%1000 == 0:
            print(i)

        curClose = tmp['Close'][i]
        futurePipReturns = tmp['PIP_Return'][i:i+interval]
        accPipFutureReturns = futurePipReturns.cumsum()

        cutEarnIndexes_long = accPipFutureReturns[accPipFutureReturns >= earnThreshold].index
        cutLossIndexes_long = accPipFutureReturns[accPipFutureReturns <= lossThreshold].index
        
        cutEarnIndexes_short = accPipFutureReturns[(accPipFutureReturns * -1) >= earnThreshold].index
        cutLossIndexes_short = accPipFutureReturns[(accPipFutureReturns * -1) <= lossThreshold].index

        hasCutEarn_long = len(cutEarnIndexes_long) > 0
        hasCutLoss_long = len(cutLossIndexes_long) > 0

        hasCutEarn_short = len(cutEarnIndexes_short) > 0
        hasCutLoss_short = len(cutLossIndexes_short) > 0

        if (hasCutEarn_long and not hasCutLoss_long) or (hasCutEarn_long and hasCutLoss_long and (cutEarnIndexes_long[0] < cutLossIndexes_long[0])):
            buyHoldSell.append(1)
            if hasCutEarn_long and hasCutLoss_long:
                simulatedReturns.append(accPipFutureReturns[0:cutLossIndexes_long[0]].max())
            else:
                simulatedReturns.append(accPipFutureReturns.max())
        elif (hasCutEarn_short and not hasCutLoss_short) or (hasCutEarn_short and hasCutLoss_short and (cutEarnIndexes_short[0] < cutLossIndexes_short[0])):
            buyHoldSell.append(-1)
            if hasCutEarn_short and hasCutLoss_short:
                simulatedReturns.append(accPipFutureReturns[0:cutLossIndexes_short[0]].min() * -1)
            else:
                simulatedReturns.append(accPipFutureReturns.min() * -1)
        else:
            buyHoldSell.append(0)
            simulatedReturns.append(0)

        # if not hasCutEarn_long and not hasCutLoss_long:
        #     lossDrawWin.append(0)
        #     simulatedReturns.append(0)
        # elif hasCutEarn_long and not hasCutLoss_long:
        #     lossDrawWin.append(1)
        #     simulatedReturns.append(accPipFutureReturns.max())
        # elif hasCutLoss_long and not hasCutEarn_long:
        #     lossDrawWin.append(-1)
        #     simulatedReturns.append(accPipFutureReturns.min())
        # else:
        #     if cutEarnIndexes_long[0] < cutLossIndexes_long[0]:
        #         lossDrawWin.append(1)
        #         simulatedReturns.append(accPipFutureReturns[0:cutLossIndexes_long[0]].max())
        #     else:
        #         lossDrawWin.append(-1)
        #         simulatedReturns.append(accPipFutureReturns[0:cutEarnIndexes_long[0]].min())
    # p.df['Loss_Draw_Win'] = lossDrawWin
    # p.df['Simulated_PIP_Returns'] = simulatedReturns

    p.df['Standardized_Volume'] = p.df['Volume']
    p.df['BuyHoldSell'] = buyHoldSell
    p.df['Simulated_PIP_Returns'] = simulatedReturns
    p.df['MinMaxScaled_Simulated_PIP_Returns'] = p.df['Simulated_PIP_Returns']

    
    import numpy as np
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    minMaxScaler = MinMaxScaler(feature_range=(0,1))
    standardizeScaler = StandardScaler()
    smoothing_window_size = 10000
    # data = p.df['MinMaxScaled_Simulated_PIP_Returns'].values.reshape(-1,1)
    # data2 = p.df['Standardized_Simulated_PIP_Returns'].values.reshape(-1,1)
    data3 = p.df['MinMaxScaled_Simulated_PIP_Returns'].values.reshape(-1,1)

    volData = p.df['Standardized_Volume'].values.reshape(-1,1)
    for di in range(0, len(data3), smoothing_window_size):
        # allPositiveReturns = data[di:di+smoothing_window_size,:][data[di:di+smoothing_window_size,0] > 0]
        # scaler.fit(allPositiveReturns)

        # minMaxScaler.fit(data[di:di+smoothing_window_size,:])
        # data[di:di+smoothing_window_size,:] = minMaxScaler.transform(data[di:di+smoothing_window_size,:])

        # standardizeScaler.fit(data2[di:di+smoothing_window_size,:])
        # data2[di:di+smoothing_window_size,:] = standardizeScaler.transform(data2[di:di+smoothing_window_size,:])

        minMaxScaler.fit(data3[di:di+smoothing_window_size,:])
        data3[di:di+smoothing_window_size,:] = minMaxScaler.transform(data3[di:di+smoothing_window_size,:])

        standardizeScaler.fit(volData[di:di+smoothing_window_size,:])
        volData[di:di+smoothing_window_size,:] = standardizeScaler.transform(volData[di:di+smoothing_window_size,:])
        volData[di:di+smoothing_window_size,:] -= volData[di:di+smoothing_window_size,:].min()

    if di+smoothing_window_size <= len(data3) - 1:
        # minMaxScaler.fit(data[di+smoothing_window_size:,:])
        # data[di+smoothing_window_size:,:] = minMaxScaler.transform(data[di+smoothing_window_size:,:])

        # standardizeScaler.fit(data2[di:di+smoothing_window_size,:])
        # data2[di:di+smoothing_window_size,:] = standardizeScaler.transform(data2[di:di+smoothing_window_size,:])

        minMaxScaler.fit(data3[di:di+smoothing_window_size,:])
        data3[di:di+smoothing_window_size,:] = minMaxScaler.transform(data3[di:di+smoothing_window_size,:])

        standardizeScaler.fit(volData[di:di+smoothing_window_size,:])
        volData[di:di+smoothing_window_size,:] = standardizeScaler.transform(volData[di:di+smoothing_window_size,:])
        volData[di:di+smoothing_window_size,:] -= volData[di:di+smoothing_window_size,:].min()

    # p.df['Standardized_Simulated_PIP_Returns'] = preprocessing.scale(p.df['Simulated_PIP_Returns'])

    # def apply_ema_smoothing(arr):
    #     # Now perform exponential moving average smoothing
    #     # So the data will have a smoother curve than the original ragged data
    #     EMA = 0.0
    #     gamma = 0.1
    #     for ti in range(len(arr)):
    #         if np.isnan(arr[ti]):
    #             continue
    #             if ti >= 1 and (not np.isnan(arr[ti-1])):
    #                 arr[ti] = arr[ti-1]
    #                 continue
    #             else:
    #                 continue
    #         EMA = gamma*arr[ti] + (1-gamma)*EMA
    #         arr[ti] = EMA
    #     return arr
    # # p.df['MinMaxScaled_Simulated_PIP_Returns'] = apply_ema_smoothing(p.df['MinMaxScaled_Simulated_PIP_Returns'].values)

    directory = path.join(dataDir, currencyPair)
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    p.save(path.join(directory, 'signals.csv'))