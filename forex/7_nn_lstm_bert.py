
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
splitByVolume = True
volumeBin = 1000
currencyPair = 'USDJPY'
resol = '1min'
rollingWindow = 3000
SEQ_LEN = 200
FUTURE_PERIOD_PREDICT = 1
BATCH_SIZE = 1024
EPOCHS = 10
RATIO_TO_PREDICT = ['MinMaxScaled_Close']

directory = path.join(dataDir, currencyPair)


#%%
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler(feature_range=(0,1))

normalizedCSV = path.join(directory, 'resol_'+resol+'-splitByVolume_'+str(splitByVolume)+'-volumeBin_'+str(volumeBin)+'_normalized.parq')
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

    minMaxScale = lambda x: minMaxScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]
    p.df['MinMaxScaled_Open'] = p.df['Open'].rolling(rollingWindow).apply(minMaxScale)
    p.df['MinMaxScaled_High'] = p.df['High'].rolling(rollingWindow).apply(minMaxScale)
    p.df['MinMaxScaled_Low'] = p.df['Low'].rolling(rollingWindow).apply(minMaxScale)
    p.df['MinMaxScaled_Close'] = p.df['Close'].rolling(rollingWindow).apply(minMaxScale)
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
test_df_start_index = last_10pct

test_df = df[(df.index >= last_10pct)]
validation_df = df[(df.index >= last_20pct) & (df.index < last_10pct)]
train_df = df[(df.index < last_20pct)]  # now the train_df is all the data up to the last 20%

train_data = train_df[RATIO_TO_PREDICT].values
valid_data = validation_df[RATIO_TO_PREDICT].values
test_data = test_df[RATIO_TO_PREDICT].values

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

weight_file = path.join(directory, 'lstm-bert-BATCH_SIZE_'+str(BATCH_SIZE)+'-SEQ_LEN_'+str(SEQ_LEN)+'-FUTURE_PERIOD_PREDICT_'+str(FUTURE_PERIOD_PREDICT)+'-resol_'+resol+'-splitByVolume_'+str(splitByVolume)+'.h5')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


nn_models = NN_Models()
lstm_bert = nn_models.get_LSTM_BERT(SEQ_LEN, FUTURE_PERIOD_PREDICT)

if not path.exists(weight_file) or isRedo:
    X_train, y_train = shuffle(X_train, y_train)
    lstm_bert.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_valid, y_valid), 
                        #callbacks = [checkpoint , lr_reduce]
                )
    # Save JSON config to disk
    # json_config = lstm_bert.to_json()
    # with open(path.join(dataDir, 'USDJPY', 'lstm-bert-model-config.json'), 'w') as json_file:
    #     json_file.write(json_config)
    # Save weights to disk
    lstm_bert.save_weights(weight_file)
else:
    lstm_bert.load_weights(weight_file)


#%% Backtesting on test set
prices = df[['Close']].values
normalized_Price = df[['MinMaxScaled_Close']].values

normalizedPrediction = lstm_bert.predict(X_test)

long_cash_balance = 0
long_holding = 0

short_cash_balance = 0
short_holding = 0

balance = [0]

positions_balance = 0
positions = []

long_count = 0
short_count = 0
pre = -1
for i in tqdm(range(0, len(normalizedPrediction))):
    predictions = normalizedPrediction[i]
    originalStartIndex = test_df_start_index + SEQ_LEN + i

    # previousSequence = np.array(prices[originalStartIndex-rollingWindow:originalStartIndex])
    # minMaxScaler.fit(previousSequence)

    # unscaledPredictions = minMaxScaler.inverse_transform([predictions])[0]
    unscaledPredictions = predictions

    curPrice = prices[originalStartIndex-1]

    isIncreasing = False
    isDecreasing = False
    if len(unscaledPredictions) > 1:
        isIncreasing = all(i < j for i,j in zip(unscaledPredictions, unscaledPredictions[1:]))
        isDecreasing = all(i > j for i,j in zip(unscaledPredictions, unscaledPredictions[1:]))
    elif len(unscaledPredictions) == 1:
        # if curPrice < unscaledPredictions[0]:
        #     isIncreasing = True
        # if curPrice > unscaledPredictions[0]:
        #     isDecreasing = True

        if pre > 0:
            if predictions[0] > pre:
                isIncreasing = True
                # isIncreasing = True
            elif predictions[0] < pre and predictions[0] > 0.2: #predictions[0]+0.02 < pre:
                isDecreasing = True
        pre = predictions[0]

    targetDirection = 0
    if isIncreasing:
        targetDirection = 1
        long_count += 1
    elif isDecreasing:
        targetDirection = -1
        short_count += 1


    if targetDirection == 1:
        positions.append({
            'Direction': 1,
            'Price': curPrice
        })
        long_cash_balance -= curPrice
        long_holding += 1

        short_cash_balance -= (short_holding * curPrice)
        short_holding = 0
    elif targetDirection == -1:
        positions.append({
            'Direction': -1,
            'Price': curPrice
        })
        long_cash_balance += (long_holding * curPrice)
        long_holding = 0

        short_cash_balance += curPrice
        short_holding += 1

    def isCut(x):
        if targetDirection == -1 and x['Direction'] == 1:
            if x['Price'] - curPrice > 0:
                return True
            elif curPrice - x['Price'] < -0.02:
                return True
        elif targetDirection == 1 and x['Direction'] == -1:
            if curPrice - x['Price'] < 0:
                return True
            elif curPrice - x['Price'] > 0.02:
                return True
        return False

    cutPositions = [x for x in positions if isCut(x)]
    positions[:] = [x for x in positions if not isCut(x)]
    for position in cutPositions:
        if position['Direction'] == 1:
            positions_balance += (curPrice - position['Price'])
        elif position['Direction'] == -1:
            positions_balance += (position['Price'] - curPrice)
    nonCloseBalance = 0
    for position in positions:
        if position['Direction'] == 1:
            nonCloseBalance += (curPrice - position['Price'])
        elif position['Direction'] == -1:
            nonCloseBalance += (position['Price'] - curPrice)

    # balance.append((long_cash_balance+(long_holding * curPrice)) + (short_cash_balance-(short_holding * curPrice)))
    # balance.append((long_cash_balance+(long_holding * curPrice)))
    # balance.append((short_cash_balance-(short_holding * curPrice)))
    balance.append(positions_balance + nonCloseBalance)

print('Long Holding & Balance:')
print(long_holding)
print(long_cash_balance)

print('--- count ---')
print('Long: '+str(long_count))
print('Short: '+str(short_count))
print('------')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Datetime')
ax1.set_ylabel(currencyPair+' Price', color=color)
ax1.plot(prices[test_df_start_index+SEQ_LEN:], label='Price', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(balance, label='Balance')
ax2.legend()

fig.tight_layout()
plt.show()
