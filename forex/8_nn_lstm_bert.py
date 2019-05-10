
#%% Initialize Global Settings
import platform
import pandas as pd
import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

isRedo = False
isIncrementalTrain = True
splitByVolume = True
volumeBin = 1000
currencyPair = 'USDJPY'
resol = '1min'
rollingWindow = 10000
SEQ_LEN = 200
FUTURE_PERIOD_PREDICT = 1
BATCH_SIZE = 1024
EPOCHS = 10
RATIO_TO_PREDICT = ['MinMaxScaled_Close']

directory = path.join(dataDir, currencyPair, 's8')


#%%
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler(feature_range=(0,1))

normalizedCSV = path.join(directory, 'resol_'+resol+'-splitByVolume_'+str(splitByVolume)+'-volumeBin_'+str(volumeBin)+'_normalized.parq')

#%%
p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))


#%%
resolutions = ['D', '6H', '3H', '1H', '30min', '15min', '10min', '5min', '1min']
multiResolFeatures = ['Open', 'High', 'Low', 'Close', 'Volume']



#%%
df = None
if not path.exists(normalizedCSV) or isRedo:
    p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))

    if splitByVolume:
        # Resample by volume
        p.df.reset_index(drop=True, inplace=True)

        cutIndexes = []
        vols = p.df['Volume'].values
        cum = 0
        for i in tqdm(range(0, len(vols))):
            cum += vols[i]
            if cum >= volumeBin:
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
            cut = cutIndexes[i]
            rows = p.df.iloc[previousIndex:cut,:]
            previousIndex = cut
            addRecord(rows)
        rows = p.df.iloc[previousIndex:,:]
        addRecord(rows)
        p.df = pd.DataFrame(results)
    else:
        # Resample by resolution
        p.set_df(p.get_df_with_resolution(resol))
        p.df.reset_index(drop=True, inplace=True)

    # p.df = p.df.iloc[0:200000,:]
    # print(p.df['Close'])

    p.df['Open-Close'] = p.df['Open'] - p.df['Close']
    p.df['High-Low'] = p.df['High'] - p.df['Low']
    p.df['Actual_Log_Return'] = np.log(p.df['Close'].shift(-1) / p.df['Close'])

    minMaxScale = lambda x: minMaxScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]
    p.df['MinMaxScaled_Open'] = p.df['Open'].rolling(rollingWindow).apply(minMaxScale)
    p.df['MinMaxScaled_High'] = p.df['High'].rolling(rollingWindow).apply(minMaxScale)
    p.df['MinMaxScaled_Low'] = p.df['Low'].rolling(rollingWindow).apply(minMaxScale)
    p.df['MinMaxScaled_Close'] = p.df['Close'].rolling(rollingWindow).apply(minMaxScale)

    p.df['MinMaxScaled_Open-Close'] = p.df['Open-Close'].rolling(rollingWindow).apply(minMaxScale)
    p.df['MinMaxScaled_High-Low'] = p.df['High-Low'].rolling(rollingWindow).apply(minMaxScale)

    p.df.dropna(inplace=True)
    p.df.reset_index(drop=True, inplace=True)
    df = p.df
    df.to_parquet(normalizedCSV)
else:
    df = pd.read_parquet(normalizedCSV)


#%% Split dataset into train, test, validate
times = sorted(df.index.values)  # get the times
last_10pct = sorted(df.index.values)[-int(0.1*len(times))]  # get the last 10% of the times
last_20pct = sorted(df.index.values)[-int(0.2*len(times))]  # get the last 20% of the times
test_df_start_index = 0

test_df = df[(df.index >= last_10pct)]
validation_df = df[(df.index >= last_20pct) & (df.index < last_10pct)]
train_df = df[(df.index < last_20pct)]  # now the train_df is all the data up to the last 20%

train_data = df[RATIO_TO_PREDICT].values
valid_data = validation_df[RATIO_TO_PREDICT].values
test_data = df[RATIO_TO_PREDICT].values

X_train = []
y_train = []
for i in range(SEQ_LEN, len(train_data)-(FUTURE_PERIOD_PREDICT-1)):
    X_train.append(train_data[i-SEQ_LEN:i])
    y_train.append(train_data[i:i+FUTURE_PERIOD_PREDICT])
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,FUTURE_PERIOD_PREDICT)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_valid = []
y_valid = []
for i in range(SEQ_LEN, len(valid_data)-(FUTURE_PERIOD_PREDICT-1)):
    X_valid.append(valid_data[i-SEQ_LEN:i])
    y_valid.append(valid_data[i:i+FUTURE_PERIOD_PREDICT])
X_valid, y_valid = np.array(X_valid), np.array(y_valid).reshape(-1,FUTURE_PERIOD_PREDICT)
# X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

X_test = []
y_test = []
for i in range(SEQ_LEN, len(test_data)-(FUTURE_PERIOD_PREDICT-1)):
    X_test.append(test_data[i-SEQ_LEN:i])
    y_test.append(test_data[i:i+FUTURE_PERIOD_PREDICT])
X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1,FUTURE_PERIOD_PREDICT)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)


#%% Construct LSTM-BERT model
import tensorflow as tf
from forex.NN_Models import NN_Models
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import ModelCheckpoint

weight_file = path.join(directory, 'lstm-bert-BATCH_SIZE_'+str(BATCH_SIZE)+'-SEQ_LEN_'+str(SEQ_LEN)+'-FUTURE_PERIOD_PREDICT_'+str(FUTURE_PERIOD_PREDICT)+'-resol_'+resol+'-splitByVolume_'+str(splitByVolume)+'.h5')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


nn_models = NN_Models()
lstm_bert = nn_models.get_LSTM_BERT(SEQ_LEN, FUTURE_PERIOD_PREDICT)

def train(X_train, y_train, X_valid, y_valid):
    X_train, y_train = shuffle(X_train, y_train)

    # filepath= path.join(directory, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    lstm_bert.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_valid, y_valid), 
                        # callbacks = [checkpoint]
                )
    lstm_bert.save_weights(weight_file)

if not path.exists(weight_file) or isRedo:
    train(X_train, y_train, X_valid, y_valid)
else:
    lstm_bert.load_weights(weight_file)
    if isIncrementalTrain:
        train(X_train, y_train, X_valid, y_valid)


#%% Backtesting on test set
# from scipy.ndimage.interpolation import shift
logReturns = df[['Actual_Log_Return']].copy().iloc[test_df_start_index+SEQ_LEN:,:]

predictionsFile = path.join(directory, 'predictions.parq')
if path.exists(predictionsFile):
    logReturns = pd.read_parquet(predictionsFile)
else:
    print('Start Predict')
    predicted = lstm_bert.predict(X_test)
    logReturns['Predicted_Normalized_Close'] = predicted.reshape(1,-1)[0]
    # logReturns['Predicted_Normalized_Close'] = np.log(logReturns['Predicted_Normalized_Close'] / logReturns['Predicted_Normalized_Close'].shift(1)))
    logReturns.to_parquet(predictionsFile)

print('Start Backtest')
def genLong(x):
    if x[1] > x[0] + 0.01:
        return 1
    elif x[1] < x[0] - 0.01:
        return -1
    else:
        return 0
logReturns['Signal'] = logReturns['Predicted_Normalized_Close'].rolling(2).apply(genLong)
# logReturns['Signal'] = np.where(logReturns['Predicted_Normalized_Close'] > (logReturns['Predicted_Normalized_Close'].shift(1)+0.1),1,-1)

longCount = len(logReturns[logReturns['Signal'] == 1].index)
shortCount = len(logReturns[logReturns['Signal'] == -1].index)
totalCount = len(logReturns.index)
print('totalCount: ' + str(totalCount))
print('longCount: ' + str(longCount) + '('+str(longCount/totalCount*100)+'%)')
print('shortCount: ' + str(shortCount) + '('+str(shortCount/totalCount*100)+'%)')

# print(logReturns['Predicted_Normalized_Close'].head(10))
# print(logReturns['Signal'].head(10))

logReturns['Strategy_Return'] = logReturns['Actual_Log_Return'] * logReturns['Signal']

actRet = logReturns['Actual_Log_Return'].cumsum()
strRet = logReturns['Strategy_Return'].cumsum()

plt.figure(figsize=(10,5))
plt.plot(actRet, color='r', label='Actual')
plt.plot(strRet, color='g', label='Strategy')
plt.legend()
plt.show()