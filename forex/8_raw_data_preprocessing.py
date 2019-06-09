
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
from scipy import stats

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

rollingWinSize = 600
batchSize = 32
epochNum = 1
weightFileName = 'raw_data_preprocessing_model.h5'
gpuId = 0
nnModleId = 2


#%% Settings Hyper Parameters and file paths
params = {
    'currency': 'USDJPY',
    'output_folder': 'outputs_using_ticks_1000_from_2005',
    'preprocessing': {
        'is_tick': True,
        'split_method': 'by_volume', # by_volume, by_time
        'volume_bin': 1000,
        'resolution': '1min'
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

raw_data_csv_path = path.join(dataDir, params['currency']+'_1MIN_(1-1-2008_31-12-2017).csv')
raw_tick_data_csv_path = path.join(dataDir, params['currency']+'_2005-1-1_2019-5-18_ticks_new.csv')

tmp_preprocess_tick_df_path1 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_1.parq')
tmp_preprocess_tick_df_path2 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_2.parq')
tmp_preprocess_tick_df_path3 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_3.parq')
tmp_preprocess_tick_df_path4 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_4.parq')

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








#%% tick statistics
df = None
if not path.exists(tmp_preprocess_tick_df_path1):
    df = pd.read_csv(raw_tick_data_csv_path)
    # df = df.iloc[0:10000000]
    df.columns = ['Timestamp', 'Bid', 'Ask', 'BidVolume', 'AskVolume']
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
    df['Time_Diff_In_Seconds'] = (df['Timestamp'] - df['Timestamp'].shift(1)).dt.total_seconds()
    df = df.round({'BidVolume': 3, 'AskVolume': 3})
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    save_df(df, tmp_preprocess_tick_df_path1)
else:
    df = read_df(tmp_preprocess_tick_df_path1)

#%% Check Bid Ask Range
if not path.exists(tmp_preprocess_tick_df_path2):
    # Financial econometric analysis at ultra-high frequency: Data
    # handling concerns
    # C.T. Brownlees∗, G.M. Gallo
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.1176&rep=rep1&type=pdf

    # A METHOD TO ‘‘CLEAN UP’’ ULTRA HIGH-FREQUENCY DATA*
    # Angelo M. Mineo**
    # Fiorella Romito**
    # file:///home/joseph/Downloads/999999_2007_0002_0036-151260.pdf

    # tmp2 = df['Bid'] - df['Bid'].shift(1)
    # bins = np.array([-0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    # bin_i = np.digitize(tmp2_val,bins,right=False)
    # unique, counts = np.unique(bin_i, return_counts=True)
    # np.asarray((unique, counts)).T
    # array([[        0,   5560038],
    #        [        1,   3323026],
    #        [        2,   8020891],
    #        [        3,   4912798],
    #        [        4,   8204253],
    #        [        5,  28597125],
    #        [        6,   7964122],
    #        [        7, 149363475],
    #        [        8,  27745821],
    #        [        9,   8218411],
    #        [       10,   5126276],
    #        [       11,   7962262],
    #        [       12,   3264986],
    #        [       13,   5706813]])

    count = 0
    k = 60
    phi = 0.002
    rollWin = k+1
    middleIndex = int(rollWin/2)
    def check_is_invalid_range1(s):
        global count
        global phi
        if count % 1000000 == 0:
            print('count: ' + str(count/1000000))
        count += 1

        middleVal = s[middleIndex]
        seriesWithoutMiddleValue = np.delete(s, (middleIndex), axis=0)

        std = np.std(seriesWithoutMiddleValue)
        mean = np.mean(seriesWithoutMiddleValue)
        
        if abs(middleVal - mean) < 3 * std + phi:
            return 0
        return 1

    df['Is_Bid_Invalid_Range'] = df['Bid'].rolling(rollWin).apply(check_is_invalid_range1, raw=True).shift(-1 * middleIndex)
    print('Bid Completed')
    count = 0

    df['Is_Ask_Invalid_Range'] = df['Ask'].rolling(rollWin).apply(check_is_invalid_range1, raw=True).shift(-1 * middleIndex)
    print('Ask Completed')

    df['Is_BidVolume_Invalid_Range'] = df['BidVolume'].rolling(rollWin).apply(check_is_invalid_range1, raw=True).shift(-1 * middleIndex)
    print('Bid Volume Completed')

    df['Is_AskVolume_Invalid_Range'] = df['AskVolume'].rolling(rollWin).apply(check_is_invalid_range1, raw=True).shift(-1 * middleIndex)
    print('Ask Volume Completed')

    save_df(df, tmp_preprocess_tick_df_path2)
else:
    df = read_df(tmp_preprocess_tick_df_path2)



#%% Data Validation
if not path.exists(tmp_preprocess_tick_df_path3):
    df['Is_Invalid_Timestamp'] = df['Time_Diff_In_Seconds'].map(lambda diff: 1 if diff <= 0 else 0)
    df['Is_Bid_Less_Than_Or_Equal_To_0'] = df['Bid'].map(lambda x: 1 if x <= 0 else 0)
    df['Is_BidVolume_Less_Than_Or_Equal_To_0'] = df['BidVolume'].map(lambda x: 1 if x <= 0 else 0)

    save_df(df, tmp_preprocess_tick_df_path3)
else:
    df = read_df(tmp_preprocess_tick_df_path3)


#%% Check Gaps
if not path.exists(tmp_preprocess_tick_df_path4):
    def check_gap(row):
        if row['Time_Diff_In_Seconds'] >= 900 and (row['Is_Bid_Invalid_Range'] == 1 or row['Is_Ask_Invalid_Range'] == 1):
            return 1
        else:
            return 0
    df['Is_Data_Gap'] = df[['Time_Diff_In_Seconds', 'Is_Bid_Invalid_Range', 'Is_Ask_Invalid_Range']].apply(check_gap, axis=1)

    # Different Methods to Clean Up Ultra High-Frequency Data(⋆)
    # http://www.old.sis-statistica.org/files/pdf/atti/rs08_spontanee_12_4.pdf
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html
    # http://www.librow.com/articles/article-7

    save_df(df, tmp_preprocess_tick_df_path4)
else:
    df = read_df(tmp_preprocess_tick_df_path4)








#%% stat
total_count = len(df.index)
invalid_bid_count = len(df[df['Is_Bid_Invalid_Range']==1].index)
invalid_ask_count = len(df[df['Is_Ask_Invalid_Range']==1].index)
invalid_bid_ask_count = len(df[(df['Is_Bid_Invalid_Range']==1) & (df['Is_Ask_Invalid_Range']==1)].index)

print('Number of invalid bid: ' + str(invalid_bid_count))
print(invalid_bid_count/total_count * 100)
print('Number of invalid ask: ' + str(invalid_ask_count))
print(invalid_ask_count/total_count * 100)
print('Number of invalid bid and ask: ' + str(invalid_bid_ask_count))
print(invalid_bid_ask_count/total_count * 100)






df[(df['Timestamp'] >= parseISODateTime('2005-07-10T23:00:00')) & (df['Timestamp'] < parseISODateTime('2005-07-10T23:01:00'))]








#%% Raw Data Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
minMaxScaler = MinMaxScaler(feature_range=(0,1))
standardizeScaler = StandardScaler()

df = None
if not path.exists(preprocessed_df_path):
    if params['preprocessing']['split_method'] == 'by_volume':
        # Resample by volume

        vol_col_name = 'Volume'
        price_col_name = 'Close'
        vol_bin = params['preprocessing']['volume_bin']
        if params['preprocessing']['is_tick']:
            df = pd.read_csv(raw_tick_data_csv_path)
            df.columns = ['Timestamp', 'Bid', 'Ask', 'BidVolume', 'AskVolume']
            df.drop(['AskVolume'], axis=1, inplace=True)

            # Convert volume to max vol_bin if larger than vol_bin to avoid duplicated timestamp
            df['BidVolume'] = df['BidVolume'].map(lambda b: vol_bin if b > vol_bin else b)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            vol_col_name = 'BidVolume'
            price_col_name = 'Bid'
        else:
            df = pd.read_csv(raw_data_csv_path)

        # numOfRecordToBeUsed = int(len(df.index) / 1.5)
        # df = df.iloc[len(df.index)-numOfRecordToBeUsed:,:]
        # df.reset_index(drop=True, inplace=True)

        #%%
        cuts = []
        vols = df[vol_col_name].values
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

        results = {
            'Timestamp': [],
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': []
        }
        def addRecord(rows):
            timestamp = rows['Timestamp'].values[-1]

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

        print('len(cuts): ')
        print(len(cuts))
        for i in tqdm(range(0, len(cuts))):
            cut = cuts[i]
            startPos = cut[0]
            endPos = cut[1]

            rows = df.iloc[startPos:endPos+1,:]
            addRecord(rows)

        print('---------')
        df = pd.DataFrame(results)
    elif params['preprocessing']['split_method'] == 'by_time':
        # Resample by resolution
        p = OHLC(raw_data_csv_path)
        p.set_df(p.get_df_with_resolution(params['preprocessing']['resolution']))
        p.df.reset_index(drop=True, inplace=True)
        df = p.df

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
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
    df['Time_Diff_Freq'] = 1 / (df['Timestamp'] - df['Timestamp'].shift(1)).dt.total_seconds()
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df['Log_Return'] = np.log(df['Close'] / df['Open']) #np.log(df['Open'] / df['Open'].shift(1))
    df['PIP_Return'] = df['Close'].diff() * 10000

    rollingWindow = params['basic_features_generation']['scaler_rolling_window']
    
    minMaxScale = lambda x: minMaxScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]
    standardScale = lambda x: standardizeScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]

    df['MinMaxScaled_Time_Diff_Freq'] = df['Time_Diff_Freq'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_Open'] = df['Open'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_High'] = df['High'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_Low'] = df['Low'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_Close'] = df['Close'].rolling(rollingWindow).apply(minMaxScale)

    df['MinMaxScaled_Open-Close'] = df['Open-Close'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_High-Low'] = df['High-Low'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_Log_Return'] = df['Log_Return'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_PIP_Return'] = df['PIP_Return'].rolling(rollingWindow).apply(minMaxScale)

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











#%% Generate Techical Indicators
if not path.exists(techical_indicators_df_path):
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['Close'], params['techical_indicators']['MACD']["fast_period"], params['techical_indicators']['MACD']["slow_period"], params['techical_indicators']['MACD']["signal_period"])
    signal = np.sign(np.sign(macdhist).diff()) # Bear: -1; Hold: 0; Bull: 1
    df['technical_MACD_short_hold_long'] = signal
    df['technical_MACD_histogram'] = macdhist

    # RSI
    real = talib.RSI(df['Close'], timeperiod=params['techical_indicators']['RSI']["timeperiod"])
    df['technical_RSI'] = real/100

    # ADX
    real = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=params['techical_indicators']['ADX']["timeperiod"])
    df['technical_ADX'] = real/100

    # STOCH (Stochastic Oscillator Slow)
    slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=params['techical_indicators']['STOCH']['fastk_period'], slowk_period=params['techical_indicators']['STOCH']['slowk_period'], slowk_matype=params['techical_indicators']['STOCH']['slowk_matype'], slowd_period=params['techical_indicators']['STOCH']['slowd_period'], slowd_matype=params['techical_indicators']['STOCH']['slowd_matype'])
    df['technical_STOCH_slowk'] = slowk / 100
    df['technical_STOCH_slowd'] = slowd / 100

    # STOCHF (Stochastic Oscillator Fast)
    fastk, fastd = talib.STOCHF(df['High'], df['Low'], df['Close'], fastk_period=params['techical_indicators']['STOCHF']['fastk_period'], fastd_period=params['techical_indicators']['STOCHF']['fastd_period'], fastd_matype=params['techical_indicators']['STOCHF']['fastd_matype'])
    df['technical_STOCHF_fastk'] = fastk / 100
    df['technical_STOCHF_fastd'] = fastd / 100

    # DX (Directional Movement Index)
    real = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=params['techical_indicators']['DX']["timeperiod"])
    df['technical_DX'] = real/100

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
    interval = params['class_labelling']['forward_looking_bars']
    simulatedReturns = []
    shortHoldLong = [] # -1 = Short; 0 = Hold; 1 = Long
    earnThreshold = 400
    lossThreshold = -100
    if params['currency'] in ['EURUSD', 'AUDUSD']:
        earnThreshold = 20
        lossThreshold = -5

    lastIndex = len(df.index)-interval-2
    for i in tqdm(range(0, len(df.index))):
        if i > lastIndex:
            shortHoldLong.append(0)
            simulatedReturns.append(0)
            continue

        # if i%1000 == 0:
        #     print(i)

        curClose = df['Close'][i]
        futurePipReturns = df['PIP_Return'][i+1:i+interval+1]
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
    df['Short_Hold_Long'] = shortHoldLong
    df['Simulated_PIP_Returns'] = simulatedReturns
    # df['MinMaxScaled_Simulated_PIP_Returns'] = df['Simulated_PIP_Returns']
    # df.to_parquet(class_label_df_path)
    save_df(df, class_label_df_path)
else: 
    # df = pd.read_parquet(class_label_df_path)
    del df
    df = read_df(class_label_df_path)

print(df[['Timestamp', 'Short_Hold_Long', 'Simulated_PIP_Returns']])
print('Number of Should-Long: ' + str(len(df[df['Short_Hold_Long']==1].index)))
print('Number of Should-Short: ' + str(len(df[df['Short_Hold_Long']==-1].index)))
print('Number of Should-Hold: ' + str(len(df[df['Short_Hold_Long']==0].index)))













#%%
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
            results[prefix+'_'+funcName] = tmp
    return pd.DataFrame(results)

resols = params['patterns_features_generation']['resampling_bars']
for resolStr in resols:
    tmp = tmp_df[['Close']].resample(resolStr, label='right').ohlc()
    patterns = detect_pattern_using_talib('patterns_'+resolStr, tmp['Close']['open'], tmp['Close']['high'], tmp['Close']['low'], tmp['Close']['close'])
    patterns['join_key'] = patterns.index
    tmp_df['join_key'] = tmp_df.index.floor(resolStr)
    tmp_df = pd.merge(tmp_df, patterns, on='join_key', how='left')
    tmp_df.index = tmp_df['Timestamp']
    tmp_df.drop('join_key', axis=1, inplace=True)


#%%

featuresToBeUsed = tmp_df.columns.values[[colName.startswith('patterns_') for colName in tmp_df.columns.values]]
tmp_df.dropna()[featuresToBeUsed].describe()



#%%
tmp_df.columns.values





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

    resols = params['patterns_features_generation']['resampling_bars']
    for resolStr in resols:
        tmp = tmp_df[['Close']].resample(resolStr, label='right').ohlc()
        patterns = detect_pattern_using_talib('patterns_'+resolStr, tmp['Close']['open'], tmp['Close']['high'], tmp['Close']['low'], tmp['Close']['close'])
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















#####################################################################
#####################################################################
#####################################################################
# Models Evaluation with Selected Features
#####################################################################
#####################################################################
#####################################################################

#%% Models Evaluation with Selected Features
print('Models Evaluation with Selected Features')
excludeColumns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open-Close', 'High-Low', 'Time_Diff_Freq', 'Log_Return', 'PIP_Return', 'Short_Hold_Long', 'Simulated_PIP_Returns']
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


# featuresToBeUsed = ['MinMaxScaled_Time_Diff_Freq', 'MinMaxScaled_Open', 'MinMaxScaled_High', 'MinMaxScaled_Low', 'MinMaxScaled_Close', 'MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low', 'MinMaxScaled_Log_Return', 'MinMaxScaled_PIP_Return', 'technical_MACD_histogram', 'technical_RSI', 'technical_ADX', 'technical_STOCH_slowk', 'technical_STOCH_slowd', 'technical_STOCHF_fastk', 'technical_DX']
# featuresToBeUsed = ['Log_Return', 'MinMaxScaled_Log_Return', 'MinMaxScaled_Time_Diff_Freq', 'MinMaxScaled_Close', 'MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low', 'technical_MACD_histogram', 'technical_RSI', 'technical_ADX', 'technical_STOCH_slowk', 'technical_STOCH_slowd', 'technical_STOCHF_fastk', 'technical_DX']

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

        times = sorted(df.index.values)  # get the times
        last_10pct = sorted(df.index.values)[-int(0.1*len(times))]  # get the last 10% of the times
        last_20pct = sorted(df.index.values)[-int(0.2*len(times))]  # get the last 20% of the times
        test_df_start_index = last_10pct

        # test_df = df[(df.index >= last_10pct)]
        # validation_df = df[(df.index >= last_20pct) & (df.index < last_10pct)]
        # train_df = df[(df.index < last_20pct)]  # now the train_df is all the data up to the last 20%

        train_df = df[(df['Timestamp'] >= parseISODateTime('2005-01-01T00:00:00')) & (df['Timestamp'] < parseISODateTime('2017-07-01T00:00:00'))]
        validation_df = df[(df['Timestamp'] >= parseISODateTime('2017-07-01T00:00:00')) & (df['Timestamp'] < parseISODateTime('2018-07-01T00:00:00'))]
        test_df = df[df['Timestamp'] >= parseISODateTime('2018-07-01T00:00:00')]

        def generate_batch_data_random(df_ref, batch_size):
            x = df_ref[FEATURES_COLS].values
            y = df_ref[TARGET_COLS].values

            df_len = len(df_ref.index)
            loopcount = (df_len-SEQ_LEN) // batch_size
            while (True):
                batchIndex = random.randint(0, loopcount-1)
                xStartIndex = SEQ_LEN + batchIndex * batch_size

                X_train = []
                y_train = []
                for i in range(xStartIndex, xStartIndex+batch_size):
                    X_train.append(x[i-SEQ_LEN:i])
                    y_train.append(y[i:i+FUTURE_PERIOD_PREDICT][0])
                yield np.array(X_train), np.array(y_train)

        # train_data = train_df[FEATURES_COLS].values
        # y_train_data = train_df[TARGET_COLS].values

        # valid_data = validation_df[FEATURES_COLS].values
        # y_valid_data = validation_df[TARGET_COLS].values

        test_data = test_df[FEATURES_COLS].values
        y_test_data = test_df[TARGET_COLS].values

        # all_data = df[FEATURES_COLS].values
        # y_all_data = df[TARGET_COLS].values

        # X_train = []
        # y_train = []
        # for i in range(SEQ_LEN, len(train_data)-(FUTURE_PERIOD_PREDICT-1)):
        #     X_train.append(train_data[i-SEQ_LEN:i])
        #     y_train.append(y_train_data[i:i+FUTURE_PERIOD_PREDICT][0])
        # X_train, y_train = np.array(X_train), np.array(y_train)
        # # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # X_valid = []
        # y_valid = []
        # for i in range(SEQ_LEN, len(valid_data)-(FUTURE_PERIOD_PREDICT-1)):
        #     X_valid.append(valid_data[i-SEQ_LEN:i])
        #     y_valid.append(y_valid_data[i:i+FUTURE_PERIOD_PREDICT][0])
        # X_valid, y_valid = np.array(X_valid), np.array(y_valid)
        # # X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

        X_test = []
        y_test = []
        for i in range(SEQ_LEN, len(test_data)-(FUTURE_PERIOD_PREDICT-1)):
            X_test.append(test_data[i-SEQ_LEN:i])
            y_test.append(y_test_data[i:i+FUTURE_PERIOD_PREDICT][0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # X_all = []
        # y_all = []
        # for i in range(SEQ_LEN, len(all_data)-(FUTURE_PERIOD_PREDICT-1)):
        #     X_all.append(all_data[i-SEQ_LEN:i])
        #     y_all.append(y_all_data[i:i+FUTURE_PERIOD_PREDICT][0])
        # X_all, y_all = np.array(X_all), np.array(y_all)
        # # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


        #%%
        # print(y_train.shape)
        # print(y_valid.shape)
        # print(y_test.shape)
        # print(y_all.shape)
        # print(X_all.shape)


        #%% Construct LSTM-BERT model
        import tensorflow as tf
        from forex.NN_Models import NN_Models
        from forex.NN_Models2 import NN_Models2
        from sklearn.utils import shuffle
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

            filepath= path.join(directory, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

            lstm_bert.fit(X_train, y_train,
                                batch_size=p['batch_size'],
                                epochs=p['epochs'],
                                validation_data=(X_valid, y_valid), 
                                # callbacks = [cp_callback],
                                callbacks = [SaveWeightCallback()],
                        )

            lstm_bert.save_weights(weightFilePath)

        def train_using_dfs(train_df, validation_df, weightFilePath):
            lstm_bert.fit_generator(generate_batch_data_random(train_df, p['batch_size']),                                                      
                                    steps_per_epoch=((len(train_df.index)-SEQ_LEN)//p['batch_size']),
                                    epochs=p['epochs'], 
                                    validation_data=generate_batch_data_random(validation_df, p['batch_size']),
                                    validation_steps=((len(validation_df.index)-SEQ_LEN)//p['batch_size']), 
                                    verbose=1,
                                    callbacks=[SaveWeightCallback()],
                                    use_multiprocessing=True,
                                    workers=10,
                                    )
            lstm_bert.save_weights(weightFilePath)

        print(len(train_df.index))
        print((len(train_df.index)-SEQ_LEN)//p['batch_size'])
        if not path.exists(weightFilePath):
            # train(X_train, y_train, X_valid, y_valid, weightFilePath)
            train_using_dfs(train_df, validation_df, weightFilePath)
        else:
            lstm_bert.load_weights(weightFilePath)
            if False:
                # train(X_train, y_train, X_valid, y_valid, weightFilePath)
                train_using_dfs(train_df, validation_df, weightFilePath)


        #%% Backtesting on test set
        # from scipy.ndimage.interpolation import shift
        logReturns = test_df[['Timestamp', 'Log_Return', 'PIP_Return', p['y_feature_column_name']]].copy().iloc[SEQ_LEN:,:]

        print('Start Predict')
        predicted = lstm_bert.predict(X_test)
        logReturns['Predicted_'+p['y_feature_column_name']] = predicted.reshape(1,-1)[0]
        # logReturns['Predicted_Return'] = np.log(logReturns['Predicted_Return'] / logReturns['Predicted_Return'].shift(1)))
        # logReturns.to_parquet(predictionsFile)

        print('Start Backtest')


        logReturns['tmp_min'] = logReturns['Log_Return'].shift(1).rolling(600).min()
        logReturns['tmp_max'] = logReturns['Log_Return'].shift(1).rolling(600).max()
        logReturns['tmp_scale'] = 1 / (logReturns['tmp_max'] - logReturns['tmp_min'] + 0.000000001)
        logReturns['Predicted_Log_Return'] = (logReturns['Predicted_'+p['y_feature_column_name']] + (logReturns['tmp_min'] * logReturns['tmp_scale'])) / logReturns['tmp_scale']


        # minMaxScaler = MinMaxScaler(feature_range=(0,1))
        # def minMaxScaleReverse(x): 
        #     minMaxScaler.fit(x.reshape(-1,1))
        # df['Predicted_Log_Return'] = logReturns['Log_Return'].rolling(200).apply(minMaxScaleReverse)



        # def genReturnLong(x):
        #     if x > 0:
        #         return 1
        #     elif x < 0:
        #         return -1
        #     else:
        #         return 0
        def genLong(x):
            if x > 0.5:
                return 1
            elif x < 0.5:
                return -1
            else:
                return 0
        logReturns['Signal'] = logReturns['Predicted_'+p['y_feature_column_name']].apply(genLong)
        # logReturns['Signal'] = logReturns['Predicted_Log_Return'].apply(genReturnLong)
        # logReturns['Signal'] = np.where(logReturns['Predicted_Return'] > (logReturns['Predicted_Return'].shift(1)+0.1),1,-1)

        longCount = len(logReturns[logReturns['Signal'] == 1].index)
        shortCount = len(logReturns[logReturns['Signal'] == -1].index)
        totalCount = len(logReturns.index)
        print('totalCount: ' + str(totalCount))
        print('longCount: ' + str(longCount) + '('+str(longCount/totalCount*100)+'%)')
        print('shortCount: ' + str(shortCount) + '('+str(shortCount/totalCount*100)+'%)')

        # print(logReturns['Predicted_Return'].head(10))
        # print(logReturns['Signal'].head(10))

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
        print(logReturns[['Predicted_'+p['y_feature_column_name'], p['y_feature_column_name'], 'Log_Return']].describe())
        print(strRet)

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