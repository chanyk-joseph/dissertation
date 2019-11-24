
#%% Initialize Global Settings
import platform
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import snappy
from fastparquet import ParquetFile
import os
import sys
import os.path as path
import random

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import talib

from forex.OHLC import OHLC
from forex.utils import *

dataDir = ''
if platform.system() == 'Windows':
    dataDir = 'F:\\modified_data'
elif platform.system() == 'Linux':
    dataDir = '/home/joseph/Desktop/datastore'
else:
    exit()

pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", '{:,.3f}'.format)

# rollingWinSize = int(sys.argv[1])
# batchSize = int(sys.argv[2])
# epochNum = int(sys.argv[3])
# weightFileName = sys.argv[4]
# gpuId = sys.argv[5]
# nnModleId = sys.argv[6]

rollingWinSize = 600
batchSize = 128
epochNum = 5
weightFileName = 'raw_data_preprocessing_model.h5'
gpuId = 0
nnModleId = '2'

#%% Settings Hyper Parameters and file paths
params = {
    'currency': 'USDJPY',
    'output_folder': 'outputs_using_ticks_1000_from_2005_test_3',
    'preprocessing': {
        'volume_bin': 1000
    },
    'basic_features_generation': {
        'scaler_rolling_window': rollingWinSize,
    },
    'techical_indicators': {
        'MACD': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        },
        'RSI': {
            'timeperiod': 14
        },
        'ADX': {
            'timeperiod': 14
        },
        'STOCH': {
            'fastk_period': 5,
            'slowk_period': 3, 
            'slowk_matype': 0, 
            'slowd_period': 3, 
            'slowd_matype': 0
        },
        'STOCHF': {
            'fastk_period': 5, 
            'fastd_period': 3, 
            'fastd_matype': 0
        },
        'DX': {
            'timeperiod': 14
        }
    },
    'patterns_features_generation': {
        'resampling_bars': ['1min', '5min', '15min', '30min', '1H', '3H', '6H', 'D']
    },
    'class_labelling': {
        'forward_looking_bars': 60
    },
    'evaluate_classification_models': False,
    'methodologies': [
        {
            'method': 'lstm-bert',
            'params': {
                'x_sequence_len': rollingWinSize,
                'y_future_sequence_len': 1,
                'batch_size': batchSize,
                'epochs': epochNum,
                'x_features_column_names': [], #['MinMaxScaled_Log_Return', 'MinMaxScaled_Close', 'MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low']],
                'y_feature_column_name': 'MinMaxScaled_Log_Return'
            }
        }
    ]
}

def hash(dictObj):
    import hashlib
    import json
    m = hashlib.md5()
    json_string = json.dumps(dictObj, sort_keys=True)
    m.update(json_string.encode('utf-8'))
    h = m.hexdigest()
    return h

raw_tick_data_df_path = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_5.parq')
output_folder = path.join(dataDir, params['currency'], params['output_folder'])
preprocessed_df_path = path.join(output_folder, 'preprocessed_'+hash(params['preprocessing'])+'.parq')

basic_features_df_path = path.join(output_folder, 'basic_features_'+hash(params['basic_features_generation'])+'.parq')
techical_indicators_df_path = path.join(output_folder, 'techical_indicators_'+hash(params['techical_indicators'])+'.parq')
class_label_df_path = path.join(output_folder, 'class_label_'+hash(params['preprocessing'])+'_'+hash(params['class_labelling'])+'.parq')
patterns_features_df_path = path.join(output_folder, 'patterns_features_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'.parq')

correlation_matrix_df_path = path.join(output_folder, 'reduced_features_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'_correlation_matrix.parq')
correlation_heat_map_before_path = path.join(output_folder, 'reduced_features_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'_correlation_matrix_before.png')
correlation_heat_map_after_path = path.join(output_folder, 'reduced_features_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'_correlation_matrix_after.png')
reduced_features_df_path = path.join(output_folder, 'reduced_features_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'.parq')

random_forest_trained_model_path = path.join(output_folder, 'random_forest_trained_weights_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'.pkl')
random_forest_importances_matrix = path.join(output_folder, 'random_forest_importances_matrix_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'.parq')
random_forest_confusion_matrix = path.join(output_folder, 'random_forest_confusion_matrix_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'.png')
random_forest_reduced_features_df_path = path.join(output_folder, 'random_forest_reduced_features_'+hash(params['preprocessing'])+'_'+hash(params['patterns_features_generation'])+'.parq')


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def save_df(df, output_file):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
def read_df(input_file):
    pf = ParquetFile(input_file)
    df = pf.to_pandas()
    return df
def add_nan_columns(df, cols):
    for col in cols:
        df[col] = np.nan






#%% Raw Data Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
minMaxScaler = MinMaxScaler(feature_range=(0,1))
standardizeScaler = StandardScaler()

df = None
if not path.exists(preprocessed_df_path):
    vol_col_name = 'Volume'
    price_col_name = 'Close'
    vol_bin = params['preprocessing']['volume_bin']

    df = read_df(raw_tick_data_df_path)
    df.drop(['Is_Bid_Invalid_Range', 'Is_Ask_Invalid_Range', 'Is_BidVolume_Invalid_Range', 'Is_AskVolume_Invalid_Range', 'Time_Diff_In_Seconds', 'Is_Invalid_Timestamp', 'Is_Bid_Less_Than_Or_Equal_To_0', 'Is_BidVolume_Less_Than_Or_Equal_To_0', 'Bid_Diff', 'Bid_Diff_Rolliing_Mean', 'Bid_Diff_Rolliing_Std', 'Abs_Bid_Diff-Bid_Diff_Rolliing_Mean'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['BidVolume'] = df['BidVolume'].map(lambda b: vol_bin if b > vol_bin else b)

    df['Dataset_ID'] = 0
    previousIndex = 0
    datasetEndIndexes = df[df['Is_Data_Gap']].index
    for k in tqdm(range(0, len(datasetEndIndexes))):
        endIndex = datasetEndIndexes[k]
        df['Dataset_ID'].where(((previousIndex > df.index) | (df.index >= endIndex)), k, inplace=True)
        previousIndex = endIndex
    df['Dataset_ID'].where((previousIndex > df.index), k+1, inplace=True) # Last Dataset


    results = {
        'Timestamp': [],
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': [],
        'Dataset_ID': []
    }
    def addRecord(rows):
        timestamp = rows['Timestamp'].values[-1]
        datasetID = rows['Dataset_ID'].values[-1]

        closes = rows[price_col_name].values
        o = closes[0]
        h = closes.max()
        l = closes.min()
        c = closes[-1]

        results['Timestamp'].append(timestamp)
        results['Open'].append(o)
        results['High'].append(h)
        results['Low'].append(l)
        results['Close'].append(c)
        results['Volume'].append(vol_bin)
        results['Dataset_ID'].append(datasetID)

    # for datasetIndex in tqdm(range(0, len(datasetEndIndexes)+1)):
    #     sub_df = df[df['Dataset_ID'] == datasetIndex].reset_index(drop=True)
    sub_df = df.reset_index(drop=True)

    vol_col_name = 'BidVolume'
    price_col_name = 'Bid'
    cuts = []
    vols = sub_df[vol_col_name].values
    cum = 0
    start_pos = 0
    for i in tqdm(range(0, len(vols))):
        if cum + vols[i] == vol_bin:
            cuts.append((start_pos, i))
            cum = 0
            start_pos = i + 1
        elif cum + vols[i] > vol_bin:
            remain = cum + vols[i] - vol_bin
            cuts.append((start_pos, i))
            start_pos = i
            
            while remain >= vol_bin:
                remain = remain - vol_bin
                cuts.append((i, i))
            if remain == 0:
                start_pos = i + 1
            cum = remain
        else:
            cum += vols[i]

    #print('len(cuts): ')
    #print(len(cuts))
    for i in tqdm(range(0, len(cuts))):
        cut = cuts[i]
        startPos = cut[0]
        endPos = cut[1]

        rows = sub_df.iloc[startPos:endPos+1,:]
        addRecord(rows)

    print('---------')
    df = pd.DataFrame(results)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # table = pa.Table.from_pandas(df)
    # pq.write_table(table, preprocessed_df_path)
    save_df(df, preprocessed_df_path)
else:
    # pf = ParquetFile(preprocessed_df_path)
    # df = pf.to_pandas()

    del df
    df = read_df(preprocessed_df_path)
    # df
df















#%% Generate Basic Features
if not path.exists(basic_features_df_path):
    rollingWindow = params['basic_features_generation']['scaler_rolling_window']        
    minMaxScale = lambda x: minMaxScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]
    standardScale = lambda x: standardizeScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]


    cols = ['Time_Diff_Freq', 'Open-Close', 'High-Low', 'Log_Return', 'PIP_Return', 
            'MinMaxScaled_Time_Diff_Freq', 
            'MinMaxScaled_Open', 'MinMaxScaled_High', 'MinMaxScaled_Low', 'MinMaxScaled_Close',
            'MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low', 'MinMaxScaled_Log_Return', 'MinMaxScaled_PIP_Return']
    add_nan_columns(df, cols)
    
    # datasetIDs = df['Dataset_ID'].unique()
    # for i, datasetID in enumerate(tqdm(datasetIDs)):
        # sub_df = df[df['Dataset_ID'] == datasetID].copy()
    sub_df = df
    sub_df_original_index = sub_df.index
    sub_df.reset_index(drop=True, inplace=True)

    # sub_df['Timestamp'] = pd.to_datetime(sub_df['Timestamp']).dt.tz_localize(None)
    sub_df.loc[:,'Time_Diff_Freq'] = 1 / ((sub_df['Timestamp'] - sub_df['Timestamp'].shift(1)).dt.total_seconds() + 0.001) # 0.001 is added to avoid dividing by 0
    sub_df.loc[:,'Open-Close'] = sub_df['Open'] - sub_df['Close']
    sub_df.loc[:,'High-Low'] = sub_df['High'] - sub_df['Low']
    sub_df.loc[:,'Log_Return'] = np.log(sub_df['Close'] / sub_df['Open']) #np.log(df['Open'] / df['Open'].shift(1))
    sub_df.loc[:,'PIP_Return'] = sub_df['Close'].diff() * 10000

    sub_df.loc[:,'MinMaxScaled_Time_Diff_Freq'] = sub_df['Time_Diff_Freq'].rolling(rollingWindow).apply(minMaxScale, raw=True)
    sub_df.loc[:,'MinMaxScaled_Open'] = sub_df['Open'].rolling(rollingWindow).apply(minMaxScale, raw=True)
    sub_df.loc[:,'MinMaxScaled_High'] = sub_df['High'].rolling(rollingWindow).apply(minMaxScale, raw=True)
    sub_df.loc[:,'MinMaxScaled_Low'] = sub_df['Low'].rolling(rollingWindow).apply(minMaxScale, raw=True)
    sub_df.loc[:,'MinMaxScaled_Close'] = sub_df['Close'].rolling(rollingWindow).apply(minMaxScale, raw=True)

    sub_df.loc[:,'MinMaxScaled_Open-Close'] = sub_df['Open-Close'].rolling(rollingWindow).apply(minMaxScale, raw=True)
    sub_df.loc[:,'MinMaxScaled_High-Low'] = sub_df['High-Low'].rolling(rollingWindow).apply(minMaxScale, raw=True)
    sub_df.loc[:,'MinMaxScaled_Log_Return'] = sub_df['Log_Return'].rolling(rollingWindow).apply(minMaxScale, raw=True)
    sub_df.loc[:,'MinMaxScaled_PIP_Return'] = sub_df['PIP_Return'].rolling(rollingWindow).apply(minMaxScale, raw=True)

        # sub_df.index = sub_df_original_index
        # df.update(sub_df)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # df.to_parquet(basic_features_df_path)
    save_df(df, basic_features_df_path)
else: 
    # df = pd.read_parquet(basic_features_df_path)

    del df
    df = read_df(basic_features_df_path)
    # df
# df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Open-Close', 'High-Low']]


#%%
df




# print(df[df['Timestamp'] >= parseISODateTime('2018-04-28T00:00:00')])
# ddddd









#%% Generate Techical Indicators
if not path.exists(techical_indicators_df_path):
    cols = ['technical_MACD_short_hold_long', 'technical_MACD_histogram',
            'technical_RSI', 'technical_ADX',
            'technical_STOCH_slowk', 'technical_STOCH_slowd',
            'technical_STOCHF_fastk', 'technical_STOCHF_fastd',
            'technical_DX']
    add_nan_columns(df, cols)

    # datasetIDs = df['Dataset_ID'].unique()
    # for i, datasetID in enumerate(tqdm(datasetIDs)):
        # sub_df = df[df['Dataset_ID'] == datasetID].copy()
    sub_df = df
    sub_df_original_index = sub_df.index
    sub_df.reset_index(drop=True, inplace=True)

    # MACD
    macd, macdsignal, macdhist = talib.MACD(sub_df['Close'], params['techical_indicators']['MACD']["fast_period"], params['techical_indicators']['MACD']["slow_period"], params['techical_indicators']['MACD']["signal_period"])
    signal = np.sign(np.sign(macdhist).diff()) # Bear: -1; Hold: 0; Bull: 1
    sub_df.loc[:,'technical_MACD_short_hold_long'] = signal
    sub_df.loc[:,'technical_MACD_histogram'] = macdhist

    # RSI
    real = talib.RSI(sub_df['Close'], timeperiod=params['techical_indicators']['RSI']["timeperiod"])
    sub_df.loc[:,'technical_RSI'] = real/100

    # ADX
    real = talib.ADX(sub_df['High'], sub_df['Low'], sub_df['Close'], timeperiod=params['techical_indicators']['ADX']["timeperiod"])
    sub_df.loc[:,'technical_ADX'] = real/100

    # STOCH (Stochastic Oscillator Slow)
    slowk, slowd = talib.STOCH(sub_df['High'], sub_df['Low'], sub_df['Close'], fastk_period=params['techical_indicators']['STOCH']['fastk_period'], slowk_period=params['techical_indicators']['STOCH']['slowk_period'], slowk_matype=params['techical_indicators']['STOCH']['slowk_matype'], slowd_period=params['techical_indicators']['STOCH']['slowd_period'], slowd_matype=params['techical_indicators']['STOCH']['slowd_matype'])
    sub_df.loc[:,'technical_STOCH_slowk'] = slowk / 100
    sub_df.loc[:,'technical_STOCH_slowd'] = slowd / 100

    # STOCHF (Stochastic Oscillator Fast)
    fastk, fastd = talib.STOCHF(sub_df['High'], sub_df['Low'], sub_df['Close'], fastk_period=params['techical_indicators']['STOCHF']['fastk_period'], fastd_period=params['techical_indicators']['STOCHF']['fastd_period'], fastd_matype=params['techical_indicators']['STOCHF']['fastd_matype'])
    sub_df.loc[:,'technical_STOCHF_fastk'] = fastk / 100
    sub_df.loc[:,'technical_STOCHF_fastd'] = fastd / 100

    # DX (Directional Movement Index)
    real = talib.DX(sub_df['High'], sub_df['Low'], sub_df['Close'], timeperiod=params['techical_indicators']['DX']["timeperiod"])
    sub_df.loc[:,'technical_DX'] = real/100

        # sub_df.index = sub_df_original_index
        # df.update(sub_df)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # df.to_parquet(techical_indicators_df_path)
    save_df(df, techical_indicators_df_path)
else: 
    # df = pd.read_parquet(techical_indicators_df_path)
    del df
    df = read_df(techical_indicators_df_path)
df












#%% Class Label Generation: Short, Hold, Long
if not path.exists(class_label_df_path):
    cols = ['Short_Hold_Long', 'Simulated_PIP_Returns']
    add_nan_columns(df, cols)

    # datasetIDs = df['Dataset_ID'].unique()
    # for i, datasetID in enumerate(tqdm(datasetIDs)):
    #     sub_df = df[df['Dataset_ID'] == datasetID].copy()
    sub_df = df
    sub_df_original_index = sub_df.index
    sub_df.reset_index(drop=True, inplace=True)

    interval = params['class_labelling']['forward_looking_bars']
    simulatedReturns = []
    shortHoldLong = [] # -1 = Short; 0 = Hold; 1 = Long
    earnThreshold = 400
    lossThreshold = -100
    if params['currency'] in ['USDCAD', 'EURUSD', 'AUDUSD']:
        earnThreshold = 20
        lossThreshold = -5

    lastIndex = len(sub_df.index)-interval-2
    for i in tqdm(range(0, len(sub_df.index))):
        if i > lastIndex:
            shortHoldLong.append(0)
            simulatedReturns.append(0)
            continue

        # if i%1000 == 0:
        #     print(i)

        curClose = sub_df['Close'][i]
        futurePipReturns = sub_df['PIP_Return'][i+1:i+interval+1]
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
            shortHoldLong.append(1)
            if hasCutEarn_long and hasCutLoss_long:
                simulatedReturns.append(accPipFutureReturns[0:cutLossIndexes_long[0]].max())
            else:
                simulatedReturns.append(accPipFutureReturns.max())
        elif (hasCutEarn_short and not hasCutLoss_short) or (hasCutEarn_short and hasCutLoss_short and (cutEarnIndexes_short[0] < cutLossIndexes_short[0])):
            shortHoldLong.append(-1)
            if hasCutEarn_short and hasCutLoss_short:
                simulatedReturns.append(accPipFutureReturns[0:cutLossIndexes_short[0]].min() * -1)
            else:
                simulatedReturns.append(accPipFutureReturns.min() * -1)
        else:
            shortHoldLong.append(0)
            simulatedReturns.append(0)
    sub_df.loc[:,'Short_Hold_Long'] = shortHoldLong
    sub_df.loc[:,'Simulated_PIP_Returns'] = simulatedReturns

        # sub_df.index = sub_df_original_index
        # df.update(sub_df)

    save_df(df, class_label_df_path)
else:
    del df
    df = read_df(class_label_df_path)

print(df[['Timestamp', 'Short_Hold_Long', 'Simulated_PIP_Returns']])
print('Number of Should-Long: ' + str(len(df[df['Short_Hold_Long']==1].index)))
print('Number of Should-Short: ' + str(len(df[df['Short_Hold_Long']==-1].index)))
print('Number of Should-Hold: ' + str(len(df[df['Short_Hold_Long']==0].index)))

























#%% Patterns Features Generation
if not path.exists(patterns_features_df_path):
    tmp_df = df.copy()
    # tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp']).dt.tz_localize(None)
    tmp_df.index = pd.DatetimeIndex(tmp_df['Timestamp'])
    def detect_pattern_using_talib(prefix, o, h, l, c):
        results = {}

        from inspect import getmembers, isfunction
        functs = [o for o in getmembers(talib) if isfunction(o[1])]
        for func in functs:
            funcName = func[0]
            if funcName.startswith('CDL'):
                print('Computing Pattern Features Using talib: ' + prefix + '_' + funcName)
                tmp = getattr(talib, funcName)(o, h, l, c) / 100
                numberOfZero = np.count_nonzero(tmp==0)
                if numberOfZero != len(tmp):
                    results[prefix+'_'+funcName] = tmp
        return pd.DataFrame(results)


    tick_df = read_df(raw_tick_data_df_path)
    tick_df.drop(['Is_Bid_Invalid_Range', 'Is_Ask_Invalid_Range', 'Is_BidVolume_Invalid_Range', 'Is_AskVolume_Invalid_Range', 'Time_Diff_In_Seconds', 'Is_Invalid_Timestamp', 'Is_Bid_Less_Than_Or_Equal_To_0', 'Is_BidVolume_Less_Than_Or_Equal_To_0', 'Bid_Diff', 'Bid_Diff_Rolliing_Mean', 'Bid_Diff_Rolliing_Std', 'Abs_Bid_Diff-Bid_Diff_Rolliing_Mean'], axis=1, inplace=True)
    tick_df.dropna(inplace=True)
    tick_df.reset_index(drop=True, inplace=True)
    tick_df.index = pd.DatetimeIndex(tick_df['Timestamp'])

    resols = params['patterns_features_generation']['resampling_bars']
    for resolStr in resols:
        tmp = tick_df[['Bid']].resample(resolStr, label='right').ohlc()
        patterns = detect_pattern_using_talib('patterns_'+resolStr, tmp['Bid']['open'], tmp['Bid']['high'], tmp['Bid']['low'], tmp['Bid']['close'])
        patterns['join_key'] = patterns.index
        tmp_df['join_key'] = tmp_df.index.floor(resolStr)
        tmp_df = pd.merge(tmp_df, patterns, on='join_key', how='left')
        tmp_df.index = tmp_df['Timestamp']
        tmp_df.drop('join_key', axis=1, inplace=True)

    print('Remove all-zero/nan features')
    summary = tmp_df.describe()
    for colName in summary.columns.values:
        col = summary[[colName]]
        if ((col.loc['min',:] == col.loc['max',:]) & (col.loc['min',:] == 0)).bool():
            tmp_df.drop(colName, axis=1, inplace=True)
            print('Dropped: '+colName)

    print('drop rows without any signals, reduce csv size')
    print('Current number of rows: ' + str(len(tmp_df.index)))
    tmp_df.dropna(inplace=True)

    m = tmp_df.iloc[:,1:].values
    mask1 = [not np.all(m[i]==0) for i in range(0, len(tmp_df.index))]
    tmp_df = tmp_df.loc[mask1,:]
    tmp_df.reset_index(drop=True, inplace=True)
    print('Drop Completed')
    print('Current number of rows: ' + str(len(tmp_df.index)))

    df = tmp_df.copy()
    # df.to_parquet(patterns_features_df_path)
    save_df(df, patterns_features_df_path)
    del tmp_df
else:
    # df = pd.read_parquet(patterns_features_df_path)
    del df
    df = read_df(patterns_features_df_path)


















#%% Dimension Reduction
if not path.exists(reduced_features_df_path):
    # Remove all-zero/nan features
    summary = df.describe()
    for colName in summary.columns.values:
        col = summary[[colName]]
        if ((col.loc['min',:] == col.loc['max',:]) & (col.loc['min',:] == 0)).bool():
            df.drop(colName, axis=1, inplace=True)
            print('Dropped: '+colName)

    # Generate correlation matrix of all columns
    corrMatrix = df.corr()
    corrMatrix.to_parquet(correlation_matrix_df_path)

    # Generate heat map to visualize correlation matrix of subset of features
    plt.clf()
    sns.heatmap(df.iloc[:,41:81].corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(20,12)
    plt.savefig(correlation_heat_map_before_path, bbox_inches='tight')


    # Remove features that has 1 correlation with previous features, i.e. remove duplication
    removedFeatureNames = []
    allCols = corrMatrix.columns.values
    for i in range(0, len(allCols)-1):
        colName_i = allCols[i]
        if colName_i in removedFeatureNames:
            continue
        for j in range(i+1, len(allCols)):
            colName_j = allCols[j]
            corr_val = corrMatrix[colName_i][colName_j]
            if corr_val == 1:
                removedFeatureNames.append(colName_j)
    print('Dropped features with correlation 1:')
    print(removedFeatureNames)
    df.drop(removedFeatureNames, axis=1, inplace=True)

    # Generate the correlation heat map again to see if there are still duplications
    plt.clf()
    sns.heatmap(df.iloc[:,41:81].corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(20,12)
    plt.savefig(correlation_heat_map_after_path, bbox_inches='tight')

    # df.to_parquet(reduced_features_df_path)
    save_df(df, reduced_features_df_path)
else:
    # df = pd.read_parquet(reduced_features_df_path)
    del df
    df = read_df(reduced_features_df_path)

























#%% Use Random Forest to find out the features importances
print('Use Random Forest to find out the features importances')
print('Number of columns: '+str(len(df.columns.values)))
print(df.columns.values)
featuresToBeUsed = df.columns.values[[colName.startswith('patterns_') or colName.startswith('technical_') for colName in df.columns.values]]
# featuresToBeUsed = np.concatenate((featuresToBeUsed, np.array(['MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low', 'MinMaxScaled_Log_Return', 'MinMaxScaled_PIP_Return'])), axis=None)
print(featuresToBeUsed)
if not path.exists(random_forest_importances_matrix):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split

    tmp = df.copy()
    tmp['Target_Prediction'] = tmp['Short_Hold_Long'].shift(-1)
    tmp.dropna(inplace=True)

    totalNumRecords = len(tmp.index)
    buyRecords = len(tmp[tmp['Target_Prediction'] == 1].index)
    sellRecords = len(tmp[tmp['Target_Prediction'] == -1].index)
    holdRecords = len(tmp[tmp['Target_Prediction'] == 0].index)

    print('Total Number of Records: ' + str(totalNumRecords))
    print('Total Number of Long Records: ' + str(buyRecords) + ' (' +str(buyRecords/totalNumRecords * 100)+ '%)')
    print('Total Number of Short Records: ' + str(sellRecords) + ' (' +str(sellRecords/totalNumRecords * 100)+ '%)')
    print('Total Number of Hold Records: ' + str(holdRecords) + ' (' +str(holdRecords/totalNumRecords * 100)+ '%)')

    train, test = train_test_split(tmp, test_size=0.2, shuffle=True)
    print('Number of records in training set: ' + str(len(train.index)))
    print('Number of records in test set: ' + str(len(test.index)))

    X_train = train[featuresToBeUsed].values
    Y_train = train['Target_Prediction'].values

    X_test = test[featuresToBeUsed].values
    Y_test = test['Target_Prediction'].values

    random_forest = None
    if not path.exists(random_forest_trained_model_path):
        # Random Forest
        random_forest = RandomForestClassifier(n_estimators=100, n_jobs=16)
        print('Start training random forest')
        random_forest.fit(X_train, Y_train)
        print('Start evaluating the test set')
        Y_prediction = random_forest.predict(X_test)
        print('Scoring the accuracy')
        random_forest.score(X_train, Y_train)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
        print(round(acc_random_forest,2,), "%")
        joblib.dump(random_forest, random_forest_trained_model_path)
    else:
        random_forest = joblib.load(random_forest_trained_model_path)

    # Calculate importance matrix from random forest
    importances = pd.DataFrame({'feature':featuresToBeUsed,'importance':np.round(random_forest.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    importances.to_parquet(random_forest_importances_matrix)

    from sklearn.metrics import accuracy_score  #for accuracy_score
    from sklearn.model_selection import KFold #for K-fold cross validation
    from sklearn.model_selection import cross_val_score #score evaluation
    from sklearn.model_selection import cross_val_predict #prediction
    from sklearn.metrics import confusion_matrix #for confusion matrix

    print('--------------Cross Check The Accuracy of the model with KFold----------------------------')
    # print('The accuracy of the Random Forest Classifier is', round(accuracy_score(Y_prediction,Y_test)*100,2))
    kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
    result_rm=cross_val_score(random_forest, train[featuresToBeUsed], train['Target_Prediction'], cv=10,scoring='accuracy')
    print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))

    #%%
    import matplotlib.pyplot as plt
    import seaborn as sns
    y_pred = cross_val_predict(random_forest, train[featuresToBeUsed], train['Target_Prediction'], cv=10)
    conf_matrix = confusion_matrix(train['Target_Prediction'], y_pred)
    sns.heatmap(conf_matrix,annot=True,fmt='3.0f',cmap="summer")
    plt.title('Confusion_matrix', y=1.05, size=15)
    plt.savefig(random_forest_confusion_matrix, bbox_inches='tight')


#%% Use only importance features according to random forest
if path.exists(random_forest_importances_matrix):
    importances = pd.read_parquet(random_forest_importances_matrix)
    importanceFeatures = importances[importances['importance']>0.01].index.values
    print('Importance features according to random forest')
    print(importanceFeatures)

    filteredFeatures = df.columns.values[[colName in importanceFeatures or not colName in featuresToBeUsed for colName in df.columns.values]]
    df = df[filteredFeatures]
    # df.to_parquet(random_forest_reduced_features_df_path)
    save_df(df, random_forest_reduced_features_df_path)






# ddddddddddddddd









#####################################################################
#####################################################################
#####################################################################
# Models Evaluation with Selected Features
#####################################################################
#####################################################################
#####################################################################

#%% Models Evaluation with Selected Features
print('Models Evaluation with Selected Features')
excludeColumns = ['Dataset_ID', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open-Close', 'High-Low', 'Time_Diff_Freq', 'Log_Return', 'PIP_Return', 'Short_Hold_Long', 'Simulated_PIP_Returns']
featuresToBeUsed = [colName for colName in df.columns.values if not colName in excludeColumns]
print('Number of columns: '+str(len(df.columns.values)))
print(df.columns.values)
print('featuresToBeUsed: ')
print(featuresToBeUsed)


if params['evaluate_classification_models']:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split

    tmp = df.copy()
    tmp['Target_Prediction'] = tmp['Short_Hold_Long'].shift(-1)
    tmp.dropna(inplace=True)

    totalNumRecords = len(tmp.index)
    buyRecords = len(tmp[tmp['Target_Prediction'] == 1].index)
    sellRecords = len(tmp[tmp['Target_Prediction'] == -1].index)
    holdRecords = len(tmp[tmp['Target_Prediction'] == 0].index)

    print('Total Number of Records: ' + str(totalNumRecords))
    print('Total Number of Long Records: ' + str(totalNumRecords) + ' (' +str(buyRecords/totalNumRecords * 100)+ '%)')
    print('Total Number of Short Records: ' + str(totalNumRecords) + ' (' +str(sellRecords/totalNumRecords * 100)+ '%)')
    print('Total Number of Hold Records: ' + str(totalNumRecords) + ' (' +str(holdRecords/totalNumRecords * 100)+ '%)')

    train, test = train_test_split(tmp, test_size=0.2, shuffle=True)
    print('Number of records in training set: ' + str(len(train.index)))
    print('Number of records in test set: ' + str(len(test.index)))

    X_train = train[featuresToBeUsed].values
    Y_train = train['Target_Prediction'].values

    X_test = test[featuresToBeUsed].values
    Y_test = test['Target_Prediction'].values


    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn import metrics


    #%% Random Forest
    random_forest = RandomForestClassifier(n_estimators=100, n_jobs=16)
    print('Start training random forest')
    random_forest.fit(X_train, Y_train)
    print('Start evaluating the test set')
    Y_prediction = random_forest.predict(X_test)
    print('Scoring the accuracy')
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(round(acc_random_forest,2,), "%")
    joblib.dump(random_forest, random_forest_trained_model_path)


    from sklearn.metrics import accuracy_score  #for accuracy_score
    from sklearn.model_selection import KFold #for K-fold cross validation
    from sklearn.model_selection import cross_val_score #score evaluation
    from sklearn.model_selection import cross_val_predict #prediction
    from sklearn.metrics import confusion_matrix #for confusion matrix

    print('--------------Cross Check The Accuracy of the model with KFold----------------------------')
    # print('The accuracy of the Random Forest Classifier is', round(accuracy_score(Y_prediction,Y_test)*100,2))
    kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
    result_rm=cross_val_score(random_forest, train[featuresToBeUsed], train['Target_Prediction'], cv=10,scoring='accuracy')
    print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))




    #%% Support Vector Machines (SVM)
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    print(round(acc_linear_svc,2,), "%")
    print(metrics.classification_report(Y_test , Y_pred))



    #%% Stochastic Gradient Descent (SGD)
    sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    sgd.score(X_train, Y_train)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    print(round(acc_sgd,2,), "%")
    print(metrics.classification_report(Y_test , Y_pred))





    #%% Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print(round(acc_log,2,), "%")
    print(metrics.classification_report(Y_test , Y_pred))




    #%% Gaussian Naïve Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    print(round(acc_gaussian,2,), "%")
    print(metrics.classification_report(Y_test , Y_pred))



    #%% Perceptron
    perceptron = Perceptron(max_iter=5)
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    print(round(acc_perceptron,2,), "%")
    print(metrics.classification_report(Y_test , Y_pred))





    #%% K-nearest Neighbours (KNN)
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    print(round(acc_knn,2,), "%")
    print(metrics.classification_report(Y_test , Y_pred))














































#%%
def genLongUsingActualLogReturn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
def genLong(x):
    if x > 0.5:
        return 1
    elif x < 0.5:
        return -1
    else:
        return 0
def Rand(start, end, num): 
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res

# featuresToBeUsed = ['MinMaxScaled_Time_Diff_Freq', 'MinMaxScaled_Open', 'MinMaxScaled_High', 'MinMaxScaled_Low', 'MinMaxScaled_Close', 'MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low', 'MinMaxScaled_Log_Return', 'MinMaxScaled_PIP_Return', 'technical_MACD_histogram', 'technical_RSI', 'technical_ADX', 'technical_STOCH_slowk', 'technical_STOCH_slowd', 'technical_STOCHF_fastk', 'technical_DX']
# featuresToBeUsed = ['Log_Return', 'MinMaxScaled_Log_Return', 'MinMaxScaled_Time_Diff_Freq', 'MinMaxScaled_Close', 'MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low', 'technical_MACD_histogram', 'technical_RSI', 'technical_ADX', 'technical_STOCH_slowk', 'technical_STOCH_slowd', 'technical_STOCHF_fastk', 'technical_DX']










#%%
output_folder = path.join(dataDir, 'EURUSD', params['output_folder'])
returndf = read_df(path.join(output_folder, 'backtest_result_epoch_5_df.parq'))
for i in range(5):
    print(returndf['Strategy_Return_'+str(i+1)].cumsum() * 100)


#%%
output_folder = path.join(dataDir, 'EURUSD', params['output_folder'])
rollingReturndf = read_df(path.join(output_folder, 'backtest_result_rolling_epoch_5_df.parq'))
for i in range(5):
    print(rollingReturndf['Strategy_Return_'+str(i+1)].cumsum() * 100)



#%%
dddddddddddd


#%%
for obj in params['methodologies']:
    h = hash(obj)
    if obj['method'] == 'lstm-bert':
        p = obj['params']
        weightFilePath = path.join(output_folder, weightFileName)

        #%% Split dataset into train, test, validate
        FEATURES_COLS = featuresToBeUsed
        if len(p['x_features_column_names']) > 0:
            FEATURES_COLS = p['x_features_column_names']
        TARGET_COLS = [p['y_feature_column_name']]
        SEQ_LEN = p['x_sequence_len']
        FUTURE_PERIOD_PREDICT = p['y_future_sequence_len']

        train_df = df[(df['Timestamp'] >= parseISODateTime('2005-01-01T00:00:00')) & (df['Timestamp'] < parseISODateTime('2017-05-01T00:00:00'))]
        validation_df = df[(df['Timestamp'] >= parseISODateTime('2017-07-01T00:00:00')) & (df['Timestamp'] < parseISODateTime('2018-07-01T00:00:00'))]
        test_df = df[df['Timestamp'] >= parseISODateTime('2018-07-01T00:00:00')]


        x_all = []
        y_all = []
        lr_all = []
        time_all = []
        # datasetIDs = sorted(df['Dataset_ID'].unique())
        # for k, datasetID in enumerate(tqdm(datasetIDs)):
        #     sub_df = df[df['Dataset_ID'] == datasetID]
        sub_df = df

        x = sub_df[FEATURES_COLS].values
        y = sub_df[TARGET_COLS].values
        lr = sub_df['Log_Return'].values
        t = sub_df['Timestamp'].values

        # if len(x) < SEQ_LEN+FUTURE_PERIOD_PREDICT:
        #     continue

        for i in range(SEQ_LEN, len(x)-(FUTURE_PERIOD_PREDICT-1)):
            x_all.append(x[i-SEQ_LEN:i])
            y_all.append(y[i:i+FUTURE_PERIOD_PREDICT][0])
            lr_all.append(lr[i])
            time_all.append(t[i])

        from sklearn.utils import shuffle
        # x_all, y_all, lr_all, time_all = shuffle(x_all, y_all, lr_all, time_all)

        # x_all = np.array(x_all)
        # y_all = np.array(y_all)
        print('len(x_all): ' + str(len(x_all)))
        print('len(y_all): ' + str(len(y_all)))

        # total_samples = len(x_all)
        # validation_start_index = int(0.8 * total_samples)
        # test_start_index = int(0.9 * total_samples)
        total_samples = len(x_all)
        validation_start_index = validation_df.index[0] - SEQ_LEN #int(0.8 * total_samples)
        test_start_index = test_df.index[0] - SEQ_LEN #int(0.9 * total_samples)

        x_train_valid = x_all[:test_start_index]
        y_train_valid = y_all[:test_start_index]

        x_train_valid, y_train_valid = shuffle(x_train_valid, y_train_valid)
        x_valid_start_i = int(0.9 * len(x_train_valid))
        x_train = x_train_valid[:x_valid_start_i]
        y_train = y_train_valid[:x_valid_start_i]
        x_valid = x_train_valid[x_valid_start_i:]
        y_valid = y_train_valid[x_valid_start_i:]

        # x_train_valid, y_train_valid = shuffle(x_train_valid, y_train_valid)
        # x_valid_start_i = int(0.8 * len(x_train_valid))
        # x_valid_end_i = int(0.9 * len(x_train_valid))
        # x_train = x_train_valid[:x_valid_start_i] + x_train_valid[x_valid_end_i:]
        # y_train = y_train_valid[:x_valid_start_i] + y_train_valid[x_valid_end_i:]
        # x_valid = x_train_valid[x_valid_start_i:x_valid_end_i]
        # y_valid = y_train_valid[x_valid_start_i:x_valid_end_i]

        # valid_indexes = random.choices(range(test_start_index), k=int(0.1 * len(x_train_valid)))
        # x_train = [x_train_valid[i] for i in range(test_start_index) if i not in valid_indexes]
        # y_train = [y_train_valid[i] for i in range(test_start_index) if i not in valid_indexes]
        # x_valid = [x_train_valid[i] for i in valid_indexes]
        # y_valid = [y_train_valid[i] for i in valid_indexes]

        x_test = np.array(x_all[test_start_index:])
        y_test = y_all[test_start_index:]
        lr_test = lr_all[test_start_index:]

        backtest_results_file = path.join(output_folder, 'backtest_results.list')
        backtest_results_rolling_file = path.join(output_folder, 'backtest_results_rolling.list')
        backtest_results = []
        backtest_results_rolling = []
        if path.exists(backtest_results_file):
            with open(backtest_results_file, 'rb') as fp:
                backtest_results = pickle.load(fp)
        if path.exists(backtest_results_rolling_file):
            with open(backtest_results_rolling_file, 'rb') as fp:
                backtest_results_rolling = pickle.load(fp)

        print('len(x_train): ' + str(len(x_train)))
        print('len(x_valid): ' + str(len(x_valid)))

        def generate_batch_data_random2(dataset, batch_size, x_all, y_all, validation_start_index, test_start_index):
            start_i = 0
            end_i = validation_start_index-1
            if dataset == 'validation':
                start_i = validation_start_index
                end_i = test_start_index-1

            while (True):
                indexesToBeUsed = Rand(start_i, end_i, batch_size)
                X = []
                Y = []
                for index in indexesToBeUsed:
                    X.append(x_all[index])
                    Y.append(y_all[index])
                yield np.array(X), np.array(Y)

        def generate_batch_data_random3(dataset, batch_size, x_all, y_all, validation_start_index, test_start_index):
            start_i = 0
            end_i = validation_start_index-1
            if dataset == 'validation':
                start_i = validation_start_index
                end_i = test_start_index-1

            totalBatches = (end_i - start_i + 1) // batch_size
            while (True):
                # batchIndex = random.randint(0, totalBatches-1)
                batchIndex = 0
                if dataset == 'train':
                    batchIndex = int((totalBatches-1) * pow(random.random(), 0.9)) # https://gamedev.stackexchange.com/questions/116832/random-number-in-a-range-biased-toward-the-low-end-of-the-range
                else:
                    batchIndex = random.randint(0, totalBatches-1)

                itemIndex = batchIndex * batch_size
                endIndex = itemIndex + batch_size

                X = x_all[itemIndex:endIndex]
                Y = y_all[itemIndex:endIndex]
                yield np.array(X), np.array(Y)

        def generate_batch_data_random4(dataset_x, dataset_y, batch_size, time_bias=False):
            totalBatches = (len(dataset_x)) // batch_size
            while (True):
                batchIndex = 0
                if time_bias:
                    batchIndex = int((totalBatches-1) * pow(random.random(), 0.8))
                else:
                    batchIndex = random.randint(0, totalBatches-1)

                itemIndex = batchIndex * batch_size
                endIndex = itemIndex + batch_size

                X = dataset_x[itemIndex:endIndex]
                Y = dataset_y[itemIndex:endIndex]
                # print('np.array(X).shape:')
                # print(np.array(X).shape)
                # print('np.array(Y).shape:')
                # print(np.array(Y).shape)
                yield np.array(X), np.array(Y)

        def generate_batch_data_random(df, batch_size):
            datasetIDs = df['Dataset_ID'].unique()
            while (True):
                datasetIDToBeUsed = datasetIDs[random.randint(0, len(datasetIDs)-1)]
                df_ref = df[df['Dataset_ID'] == datasetIDToBeUsed]

                x = df_ref[FEATURES_COLS].values
                y = df_ref[TARGET_COLS].values
                # tmp = np.where(y > 0, 2, y)
                # tmp = np.where(tmp < 0, 1, tmp)
                # y = to_categorical(tmp)
                df_len = len(df_ref.index)
                loopcount = (df_len-SEQ_LEN) // batch_size
                if loopcount < 1:
                    continue

                batchIndex = random.randint(0, loopcount-1)
                xStartIndex = SEQ_LEN + batchIndex * batch_size

                X_train = []
                y_train = []
                for i in range(xStartIndex, xStartIndex+batch_size):
                    X_train.append(x[i-SEQ_LEN:i])
                    y_train.append(y[i:i+FUTURE_PERIOD_PREDICT][0])
                yield np.array(X_train), np.array(y_train)


        #%% Construct LSTM-BERT model
        import tensorflow as tf
        from forex.NN_Models import NN_Models
        from forex.NN_Models2 import NN_Models2
        from tensorflow.python.keras.callbacks import ModelCheckpoint

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuId
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)


        nn_models = None
        lstm_bert = None
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=1, save_best_only=True)
        if nnModleId == '2':
            nn_models = NN_Models2()
            lstm_bert = nn_models.get_LSTM_BERT((SEQ_LEN, len(FEATURES_COLS)), FUTURE_PERIOD_PREDICT)
        else:
            nn_models = NN_Models()
            lstm_bert = nn_models.get_LSTM_BERT((SEQ_LEN, len(FEATURES_COLS)), FUTURE_PERIOD_PREDICT)


        class SaveWeightCallback(tf.keras.callbacks.Callback):
            def on_train_batch_begin(self, batch, logs=None):
                return
            def on_train_batch_end(self, batch, logs=None):
                return
            def on_test_batch_begin(self, batch, logs=None):
                return
            def on_test_batch_end(self, batch, logs=None):
                lstm_bert.save_weights(weightFilePath)
                return

        def train(X_train, y_train, X_valid, y_valid, weightFilePath):
            X_train, y_train = shuffle(X_train, y_train)

            lstm_bert.fit(X_train, y_train,
                                batch_size=p['batch_size'],
                                epochs=p['epochs'],
                                validation_data=(X_valid, y_valid), 
                                # callbacks = [cp_callback],
                                callbacks = [SaveWeightCallback()],
                        )

            lstm_bert.save_weights(weightFilePath)

        def backtest(lstm_bert, x_test, y_test, lr_test, backtest_results, backtest_results_rolling):
            global x_all, y_all, x_train_valid, y_train_valid, test_start_index, output_folder, weightFileName, weightFilePath, genLong
            print('Backtesting on test set')
            
            predicted = lstm_bert.predict(x_test)
            predicted = predicted.reshape(1,-1)[0]
            predicted = np.array(list(map(genLong, predicted)))
            backtest_results.append(predicted)

            proposedPngFileName = ""
            for i in range(9999):
                proposedPngFileName = path.join(output_folder, 'backtest_result_epoch_' + str(i+1) + '.png')
                if not path.exists(proposedPngFileName):
                    break

            plt.figure(figsize=(10,5))
            returns_df = pd.DataFrame({'Log_Return': lr_test})
            actRet = returns_df['Log_Return'].cumsum() * 100
            plt.plot(actRet, color='k', label='Actual Log Return Of '+params['currency'])
            for i, epochPrediction in enumerate(backtest_results):
                epochNum = i+1
                returns_df['Strategy_Return_'+str(epochNum)] = returns_df['Log_Return'] * epochPrediction
                strRet = returns_df['Strategy_Return_'+str(epochNum)].cumsum() * 100
                plt.plot(strRet, label='Strategy Accumulated Log Return | Epoch: ' + str(epochNum))

            save_df(returns_df, path.join(output_folder, 'backtest_result_epoch_' + str(epochNum) + '_df.parq'))

            plt.legend()
            plt.savefig(proposedPngFileName)




            # Predict and Train Iteratively
            lstm_bert.save_weights(weightFilePath)
            weightFilePathTmp = path.join(output_folder, 'tmp_'+weightFileName)
            print('len(x_test): ' + str(len(x_test)))
            s = 128
            numBatches = len(x_test) // s
            resultPredict = np.array([])
            x_train_shuffled, y_train_shuffled = shuffle(x_all[0:test_start_index], y_all[0:test_start_index])
            for batchId in tqdm(range(numBatches)):
                startIndex = batchId * s
                endIndex = startIndex + s

                sub_x = x_test[startIndex:endIndex]
                sub_y = y_test[startIndex:endIndex]
                print('len(sub_x): ' + str(len(sub_x)))
                # # all_sub_x = x_test[0:endIndex]
                # # all_sub_y = y_test[0:endIndex]

                # # tsi = startIndex
                # # if tsi - s >= 0:
                # #     tsi = tsi - s
                # tsi = 0
                # x_test_roll = x_test[tsi:endIndex]
                # y_test_roll = y_test[tsi:endIndex]
                # numOfPastSamples = int(len(x_test_roll) / 0.25) - len(x_test_roll)
                # totalBatches = int(test_start_index // numOfPastSamples)
                # bid = random.randint(0, totalBatches-1)
                # si = bid * numOfPastSamples
                # ei = si + numOfPastSamples
                # all_sub_x = np.concatenate((np.array(x_train_valid[si:ei]), x_test_roll))
                # all_sub_y = y_train_valid[si:ei] + y_test_roll
                # all_sub_x, all_sub_y = shuffle(all_sub_x, all_sub_y)
                # # all_sub_x = np.array(all_sub_x)
                # print(len(all_sub_x))
                # print(len(all_sub_y))

                # si = test_start_index - int(endIndex / 0.3)
                # ei = test_start_index + endIndex
                # all_sub_x = x_all[si:ei]
                # all_sub_y = y_all[si:ei]
                # all_sub_x, all_sub_y = shuffle(all_sub_x, all_sub_y)
                # all_sub_x = np.array(all_sub_x)

                rollingSize = 10240
                numOfPastSamples = int(rollingSize / 0.75) - rollingSize
                totalBatches = int(test_start_index // numOfPastSamples)
                bid = random.randint(0, totalBatches-1)
                si = bid * numOfPastSamples
                ei = si + numOfPastSamples

                all_sub_x = x_train_shuffled[si:ei] + x_all[(test_start_index+endIndex-rollingSize):(test_start_index+endIndex)]
                all_sub_y = y_train_shuffled[si:ei] + y_all[(test_start_index+endIndex-rollingSize):(test_start_index+endIndex)]
                # si = test_start_index - numOfPastSamples
                # ei = test_start_index + endIndex
                # all_sub_x = x_all[si:ei]
                # all_sub_y = y_all[si:ei]
                all_sub_x, all_sub_y = shuffle(all_sub_x, all_sub_y)
                print(len(all_sub_x))
                # print(len(all_sub_y))

                predicted = lstm_bert.predict(sub_x)
                print('len(predicted.reshape(1,-1)[0]): ' + str(len(predicted.reshape(1,-1)[0])))
                resultPredict = np.concatenate((resultPredict, predicted.reshape(1,-1)[0]))
                lstm_bert.fit_generator(generate_batch_data_random4(all_sub_x, all_sub_y, p['batch_size']), 
                                        steps_per_epoch=int((len(all_sub_x) // p['batch_size']) / 10),
                                        epochs=1, 
                                        verbose=1,
                                        use_multiprocessing=True,
                                        workers=5,
                                        )

            lstm_bert.save_weights(weightFilePathTmp)
            remain = len(x_test) - numBatches * s
            startIndex = numBatches * s
            sub_x = x_test[startIndex:]
            sub_y = y_test[startIndex:]
            predicted = lstm_bert.predict(sub_x)
            resultPredict = np.concatenate((resultPredict, predicted.reshape(1,-1)[0]))
            predicted = np.array(list(map(genLong, resultPredict)))
            backtest_results_rolling.append(predicted)
            lstm_bert.load_weights(weightFilePath)

            proposedPngFileName = ""
            for i in range(9999):
                proposedPngFileName = path.join(output_folder, 'backtest_result_rolling_epoch_' + str(i+1) + '.png')
                if not path.exists(proposedPngFileName):
                    break

            plt.figure(figsize=(10,5))
            returns_df = pd.DataFrame({'Log_Return': lr_test})
            actRet = returns_df['Log_Return'].cumsum() * 100
            plt.plot(actRet, color='k', label='Actual Log Return Of '+params['currency'])
            for i, epochPrediction in enumerate(backtest_results_rolling):
                epochNum = i+1
                returns_df['Strategy_Return_'+str(epochNum)] = returns_df['Log_Return'] * epochPrediction
                strRet = returns_df['Strategy_Return_'+str(epochNum)].cumsum() * 100
                plt.plot(strRet, label='Strategy Accumulated Log Return | Epoch: ' + str(epochNum))

            save_df(returns_df, path.join(output_folder, 'backtest_result_rolling_epoch_' + str(epochNum) + '_df.parq'))

            plt.legend()
            plt.savefig(proposedPngFileName)


        def train_using_dfs(train_df, validation_df, weightFilePath):
            print('Start Training')
            global x_train, y_train, x_valid, y_valid, x_train_valid, y_train_valid, x_valid_start_i
            # lstm_bert.fit_generator(generate_batch_data_random(train_df, p['batch_size']),                                                      
            #                         steps_per_epoch=((len(train_df.index)-SEQ_LEN)//p['batch_size']),
            #                         epochs=p['epochs'], 
            #                         validation_data=generate_batch_data_random(validation_df, p['batch_size']),
            #                         validation_steps=((len(validation_df.index)-SEQ_LEN)//p['batch_size']), 
            #                         verbose=1,
            #                         callbacks=[SaveWeightCallback()],
            #                         use_multiprocessing=True,
            #                         workers=5,
            #                         )

            # lstm_bert.fit_generator(generate_batch_data_random3('train', p['batch_size'], x_all, y_all, validation_start_index, test_start_index),                                                      
            #                         steps_per_epoch=((validation_start_index - 1)//p['batch_size']),
            #                         epochs=p['epochs'], 
            #                         validation_data=generate_batch_data_random3('validation', p['batch_size'], x_all, y_all, validation_start_index, test_start_index),
            #                         validation_steps=((test_start_index - validation_start_index)//p['batch_size']), 
            #                         verbose=1,
            #                         callbacks=[SaveWeightCallback()],
            #                         use_multiprocessing=True,
            #                         workers=5,
            #                         )

            for i in range(p['epochs']):
                lstm_bert.fit_generator(generate_batch_data_random4(x_train, y_train, p['batch_size']),                                                      
                                        steps_per_epoch=(len(x_train)//p['batch_size']),
                                        epochs=1, 
                                        validation_data=generate_batch_data_random4(x_valid, y_valid, p['batch_size']),
                                        validation_steps=(len(x_valid)//p['batch_size']), 
                                        verbose=1,
                                        # callbacks=[SaveWeightCallback()],
                                        use_multiprocessing=True,
                                        workers=5,
                                        )
                lstm_bert.save_weights(path.join(output_folder, str(i)+'_'+weightFileName))
                lstm_bert.save_weights(weightFilePath)
                
                x_train_valid, y_train_valid = shuffle(x_train_valid, y_train_valid)
                x_valid_start_i = int(0.9 * len(x_train_valid))
                x_train = x_train_valid[:x_valid_start_i]
                y_train = y_train_valid[:x_valid_start_i]
                x_valid = x_train_valid[x_valid_start_i:]
                y_valid = y_train_valid[x_valid_start_i:]

                backtest(lstm_bert, x_test, y_test, lr_test, backtest_results, backtest_results_rolling)

            print('Completed Training')
            # lstm_bert.save_weights(weightFilePath)
            # print('Saved weight')

        # print(len(train_df.index))
        # print((len(train_df.index)-SEQ_LEN)//p['batch_size'])
        if not path.exists(weightFilePath):
            # train(X_train, y_train, X_valid, y_valid, weightFilePath)
            train_using_dfs(train_df, validation_df, weightFilePath)
            # train(np.array(x_train), np.array(y_train), np.array(x_valid), np.array(y_valid), weightFilePath)
        else:
            lstm_bert.load_weights(weightFilePath)
            if True:
                # train(X_train, y_train, X_valid, y_valid, weightFilePath)
                train_using_dfs(train_df, validation_df, weightFilePath)
                # train(np.array(x_train), np.array(y_train), np.array(x_valid), np.array(y_valid), weightFilePath)

        #backtest(lstm_bert, x_test, y_test, lr_test, backtest_results, backtest_results_rolling)

        with open(backtest_results_file, 'wb') as fp:
            pickle.dump(backtest_results, fp)




        ddddd
        # x_test_all = np.array(x_all)
        # y_test_all = y_all
        # lr_test_all = lr_all
        x_test = np.array(x_all[test_start_index:])
        y_test = y_all[test_start_index:]
        lr_test = lr_all[test_start_index:]
 

        # ddd

        # Predict and Train Iteratively
        weightFilePathTmp = path.join(output_folder, 'tmp_'+weightFileName)
        print('len(x_test): ' + str(len(x_test)))
        s = 256
        numBatches = len(x_test) // s
        resultPredict = np.array([])
        for batchId in tqdm(range(numBatches)):
            startIndex = batchId * s
            endIndex = startIndex + s

            sub_x = x_test[startIndex:endIndex]
            sub_y = y_test[startIndex:endIndex]

            all_sub_x = x_test[0:endIndex]
            all_sub_y = y_test[0:endIndex]

            predicted = lstm_bert.predict(sub_x)
            print('len(sub_x): ' + str(len(sub_x)))
            print('len(predicted.reshape(1,-1)[0]): ' + str(len(predicted.reshape(1,-1)[0])))
            resultPredict = np.concatenate((resultPredict, predicted.reshape(1,-1)[0]))
            lstm_bert.fit_generator(generate_batch_data_random4(all_sub_x, all_sub_y, p['batch_size'], True), 
                                    steps_per_epoch=30,
                                    epochs=1, 
                                    verbose=1,
                                    use_multiprocessing=True,
                                    workers=2,
                                    )

            all_sub_lr = lr_test[0:endIndex]
            returns_df = pd.DataFrame({'Log_Return': all_sub_lr})
            returns_df['Signal'] = resultPredict
            returns_df['Signal'] = returns_df['Signal'].apply(genLong)
            returns_df['Strategy_Return'] = returns_df['Log_Return'] * returns_df['Signal']
            actRet = returns_df['Log_Return'].cumsum() * 100
            strRet = returns_df['Strategy_Return'].cumsum() * 100
            print('actRet: ' + str(actRet))
            print('strRet: ' + str(strRet))

        print('Completed Training')
        lstm_bert.save_weights(weightFilePathTmp)
        remain = len(x_test) - numBatches * s
        startIndex = numBatches * s
        sub_x = x_test[startIndex:]
        sub_y = y_test[startIndex:]
        predicted = lstm_bert.predict(sub_x)
        resultPredict = np.concatenate((resultPredict, predicted.reshape(1,-1)[0]))

        returns_df = pd.DataFrame({'Log_Return': lr_test})
        returns_df['Signal'] = resultPredict
        returns_df['Signal'] = returns_df['Signal'].apply(genLong)
        returns_df['Strategy_Return'] = returns_df['Log_Return'] * returns_df['Signal']
        actRet = returns_df['Log_Return'].cumsum() * 100
        strRet = returns_df['Strategy_Return'].cumsum() * 100

        plt.figure(figsize=(10,5))
        plt.plot(actRet, color='r', label='Actual Log Return Of '+params['currency'])
        plt.plot(strRet, color='g', label='Strategy Accumulated Log Return')
        plt.legend()
        plt.savefig(path.join(output_folder, 'rolling-train-return.png'))
        # plt.show()
        # dddd




        # predicted = lstm_bert.predict(x_test)
        # predicted = predicted.reshape(1,-1)[0]
        # predicted = np.array(list(map(genLong, predicted)))

        # returns_df = pd.DataFrame({'Log_Return': lr_test})
        # returns_df['Signal'] = predicted
        # # returns_df['Signal'] = returns_df['Signal'].apply(genLong)

        # returns_df['Strategy_Return'] = returns_df['Log_Return'] * returns_df['Signal']
        # actRet = returns_df['Log_Return'].cumsum() * 100
        # strRet = returns_df['Strategy_Return'].cumsum() * 100

        # plt.figure(figsize=(10,5))
        # plt.plot(actRet, color='r', label='Actual Log Return Of '+params['currency'])
        # plt.plot(strRet, color='g', label='Strategy Accumulated Log Return')
        # plt.legend()
        # plt.show()
        ddddddddddddddd





        test_df = test_df.copy()
        test_df['Signal'] = 0 # -1=Short; 0=No Action; 1=Long
        datasetIDs = test_df['Dataset_ID'].unique()
        #for i, datasetID in enumerate(tqdm(datasetIDs)):
        sub_df = test_df
        #print(len(sub_df.index))
        #if len(sub_df.index) < SEQ_LEN+FUTURE_PERIOD_PREDICT:
        #    continue

        #print(len(sub_df.index))
        test_data = sub_df[FEATURES_COLS].values
        y_test_data = sub_df[TARGET_COLS].values
        X_test = []
        y_test = []
        for i in range(SEQ_LEN, len(test_data)-(FUTURE_PERIOD_PREDICT-1)):
            X_test.append(test_data[i-SEQ_LEN:i])
            y_test.append(y_test_data[i:i+FUTURE_PERIOD_PREDICT][0])
        X_test, y_test = np.array(X_test), np.array(y_test)

        startIndex = sub_df.index[SEQ_LEN]
        tmp_df = sub_df[sub_df.index >= startIndex].copy()
        predicted = lstm_bert.predict(X_test)
        tmp_df['Signal'] = predicted.reshape(1,-1)[0]
        tmp_df['Signal'] = tmp_df['Signal'].apply(genLong)
        test_df.update(tmp_df)

        logReturns = test_df[['Timestamp', 'Log_Return', 'PIP_Return', p['y_feature_column_name'], 'Signal']].copy()
        longCount = len(logReturns[logReturns['Signal'] == 1].index)
        shortCount = len(logReturns[logReturns['Signal'] == -1].index)
        totalCount = len(logReturns.index)
        print('totalCount: ' + str(totalCount))
        print('longCount: ' + str(longCount) + '('+str(longCount/totalCount*100)+'%)')
        print('shortCount: ' + str(shortCount) + '('+str(shortCount/totalCount*100)+'%)')


        logReturns['Strategy_Return'] = logReturns['Log_Return'] * logReturns['Signal']
        actRet = logReturns['Log_Return'].cumsum() * 100
        strRet = logReturns['Strategy_Return'].cumsum() * 100

        plt.figure(figsize=(10,5))
        plt.plot(actRet, color='r', label='Actual Log Return Of '+params['currency'])
        plt.plot(strRet, color='g', label='Strategy Accumulated Log Return')
        plt.legend()
        plt.show()


        #%%
        correctLong = len(logReturns[logReturns['Signal'] == 1][logReturns['Log_Return'] > 0].index)
        wrongLong = len(logReturns[logReturns['Signal'] == 1][logReturns['Log_Return'] < 0].index)
        print('Total Long Signal: '+str(longCount))
        if longCount > 0:
            print('Correct: '+str(correctLong) + '('+str(correctLong/longCount*100)+'%)')
            print('Wrong: '+str(wrongLong) + '('+str(wrongLong/longCount*100)+'%)')

        correctShort = len(logReturns[logReturns['Signal'] == -1][logReturns['Log_Return'] < 0].index)
        wrongShort = len(logReturns[logReturns['Signal'] == -1][logReturns['Log_Return'] > 0].index)
        print('Total Short Signal: '+str(shortCount))
        if shortCount > 0:
            print('Correct: '+str(correctShort) + '('+str(correctShort/shortCount*100)+'%)')
            print('Wrong: '+str(wrongShort) + '('+str(wrongShort/shortCount*100)+'%)')


        #%%
        from sklearn import metrics
        logReturns['Actual_Signal'] = logReturns[p['y_feature_column_name']].apply(genLong)
        # logReturns['Actual_Signal'] = logReturns['Log_Return'].apply(genReturnLong)
        print(metrics.confusion_matrix(logReturns['Actual_Signal'] , logReturns['Signal']))
        print(metrics.classification_report(logReturns['Actual_Signal'] , logReturns['Signal']))

        # double check
        longEarn = logReturns[(logReturns['Log_Return'] > 0) & (logReturns['Predicted_'+p['y_feature_column_name']] > 0.5)]['Log_Return'].cumsum().iloc[-1]
        longLoss = logReturns[(logReturns['Log_Return'] < 0) & (logReturns['Predicted_'+p['y_feature_column_name']] > 0.5)]['Log_Return'].cumsum().iloc[-1]

        shortEarn = logReturns[(logReturns['Log_Return'] < 0) & (logReturns['Predicted_'+p['y_feature_column_name']] < 0.5)]['Log_Return'].cumsum().iloc[-1]
        shortLoss = logReturns[(logReturns['Log_Return'] > 0) & (logReturns['Predicted_'+p['y_feature_column_name']] < 0.5)]['Log_Return'].cumsum().iloc[-1]

        print(longEarn)
        print(longLoss)
        print(shortEarn)
        print(shortLoss)

        expectedVal = abs(longEarn) + abs(shortEarn) - abs(longLoss) - abs(shortLoss)
        print('expectedVal: ')
        print(expectedVal)

sys.exit()