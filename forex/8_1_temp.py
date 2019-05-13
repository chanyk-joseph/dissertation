
#%% Initialize Global Settings
import platform
import pandas as pd
import os
import sys
import os.path as path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from forex.OHLC import OHLC

dataDir = ''
if platform.system() == 'Windows':
    dataDir = 'F:\\modified_data'
elif platform.system() == 'Linux':
    dataDir = '/home/joseph/Desktop/datastore'
else:
    exit()

pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", '{:,.3f}'.format)


#%% Settings Hyper Parameters and file paths
params = {
    'currency': sys.argv[1],
    'output_folder': 'outputs',
    'preprocessing': {
        'split_method': 'by_volume', # by_volume, by_time
        'volume_bin': 500,
        'resolution': '1min'
    },
    'features_generation': {
        'rolling_window_for_min_max_scaler': 200,
# resolutions = ['D', '6H', '3H', '1H', '30min', '15min', '10min', '5min', '1min']
# multiResolFeatures = ['Open', 'High', 'Low', 'Close', 'Volume']
    },
    'class_labelling': {
        'forward_looking_bars': 60
    },
    'methodologies': [
        {
            'method': 'lstm-bert',
            'params': {
                'x_sequence_len': 200,
                'y_future_sequence_len': 1,
                'batch_size': 1024,
                'epochs': 10,
                'x_features_column_names': ['MinMaxScaled_Return', 'MinMaxScaled_Close', 'MinMaxScaled_Open-Close', 'MinMaxScaled_High-Low'],
                'y_feature_column_name': 'MinMaxScaled_Return'
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
output_folder = path.join(dataDir, params['currency'], params['output_folder'])
preprocessed_df_path = path.join(output_folder, 'preprocessed_'+hash(params['preprocessing'])+'.parq')

basic_features_df_path = path.join(output_folder, 'basic_features_'+hash(params['features_generation'])+'.parq')
class_label_df_path = path.join(output_folder, 'class_label_'+hash(params['preprocessing'])+'_'+hash(params['class_labelling'])+'.parq')


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#%% Raw Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler(feature_range=(0,1))

df = None
if not path.exists(preprocessed_df_path):
    if params['preprocessing']['split_method'] == 'by_volume':
        # Resample by volume
        df = pd.read_csv(raw_data_csv_path)
        # df = df.iloc[0:100,:]

        cutIndexes = []
        vols = df['Volume'].values
        cum = 0
        for i in tqdm(range(0, len(vols))):
            cum += vols[i]
            if cum >= params['preprocessing']['volume_bin']:
                cutIndexes.append(i)
                cum = 0
        
        results = {
            'Timestamp': [],
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': []
        }
        previousIndex = 0
        def addRecord(rows):
            timestamp = rows['Timestamp'].values[-1]
            cumVolume = rows['Volume'].sum()
            
            closes = rows['Close'].values
            o = closes[0]
            h = closes.max()
            l = closes.min()
            c = closes[-1]

            results['Timestamp'].append(timestamp)
            results['Open'].append(o)
            results['High'].append(h)
            results['Low'].append(l)
            results['Close'].append(c)
            results['Volume'].append(cumVolume)
        for i in tqdm(range(0, len(cutIndexes))):
            cut = cutIndexes[i]+1
            rows = df.iloc[previousIndex:cut,:]
            previousIndex = cut
            addRecord(rows)
        rows = df.iloc[previousIndex:,:]
        if len(rows.index) > 0:
            addRecord(rows)
        df = pd.DataFrame(results)
    elif params['preprocessing']['split_method'] == 'by_time':
        # Resample by resolution
        p = OHLC(raw_data_csv_path)
        p.set_df(p.get_df_with_resolution(params['preprocessing']['resolution']))
        p.df.reset_index(drop=True, inplace=True)
        df = p.df

    print(df.head())

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(preprocessed_df_path)
else:
    df = pd.read_parquet(preprocessed_df_path)
df

#%% Generate Basic Features
if not path.exists(basic_features_df_path):
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df['Log_Return'] = np.log(df['Open'] / df['Open'].shift(1))
    df['PIP_Return'] = df['Close'].diff() * 10000

    rollingWindow = params['features_generation']['rolling_window_for_min_max_scaler']
    minMaxScale = lambda x: minMaxScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]
    df['MinMaxScaled_Open'] = df['Open'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_High'] = df['High'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_Low'] = df['Low'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_Close'] = df['Close'].rolling(rollingWindow).apply(minMaxScale)

    df['MinMaxScaled_Open-Close'] = df['Open-Close'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_High-Low'] = df['High-Low'].rolling(rollingWindow).apply(minMaxScale)
    df['MinMaxScaled_Log_Return'] = df['Log_Return'].rolling(rollingWindow).apply(minMaxScale)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(basic_features_df_path)
else: 
    df = pd.read_parquet(basic_features_df_path)
df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Open-Close', 'High-Low']]


#%% Class Label Generation: Short, Hold, Long
class_label_df_path
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
    for i in range(0, len(df.index)):
        if i > lastIndex:
            shortHoldLong.append(0)
            simulatedReturns.append(0)
            continue

        if i%1000 == 0:
            print(i)

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
    # df['MinMaxScaled_Simulated_PIP_Returns'] = p.df['Simulated_PIP_Returns']
    df.to_parquet(class_label_df_path)
else: 
    df = pd.read_parquet(class_label_df_path)

print(df[['Timestamp', 'Short_Hold_Long', 'Simulated_PIP_Returns']])
print('Number of Should-Long: ' + str(len(df[df['Short_Hold_Long']==1].index)))
print('Number of Should-Short: ' + str(len(df[df['Short_Hold_Long']==-1].index)))
print('Number of Should-Hold: ' + str(len(df[df['Short_Hold_Long']==0].index)))









