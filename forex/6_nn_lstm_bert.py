
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


#%%
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler(feature_range=(0,1))

p = OHLC(path.join(dataDir, 'USDJPY_1MIN_(1-1-2008_31-12-2017).csv'))
p.set_df(p.get_df_with_resolution('5min'))

p.df.reset_index(drop=True, inplace=True)
# p.df = p.df.iloc[0:200000,:]

print(p.df['Close'])
minMaxScale = lambda x: minMaxScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]
rollingWindow = 30000
p.df['MinMaxScaled_Open'] = p.df['Open'].rolling(rollingWindow).apply(minMaxScale)
p.df['MinMaxScaled_High'] = p.df['High'].rolling(rollingWindow).apply(minMaxScale)
p.df['MinMaxScaled_Low'] = p.df['Low'].rolling(rollingWindow).apply(minMaxScale)
p.df['MinMaxScaled_Close'] = p.df['Close'].rolling(rollingWindow).apply(minMaxScale)


#%%
# p.df['MinMaxScaled_Close'].plot()

#%%
import numpy as np

RATIO_TO_PREDICT = ['MinMaxScaled_Close']
SEQ_LEN = 200
FUTURE_PERIOD_PREDICT = 1
BATCH_SIZE = 1024
EPOCHS = 10

# df = p.df.copy().dropna(inplace=True)

#%%
p.df.dropna(inplace=True)
p.df.reset_index(drop=True, inplace=True)
p.df

#%%
df = p.df

#%%
times = sorted(df.index.values)  # get the times
last_10pct = sorted(df.index.values)[-int(0.1*len(times))]  # get the last 10% of the times
last_20pct = sorted(df.index.values)[-int(0.2*len(times))]  # get the last 20% of the times

test_df = df[(df.index >= last_10pct)]
validation_df = df[(df.index >= last_20pct) & (df.index < last_10pct)]
train_df = df[(df.index < last_20pct)]  # now the train_df is all the data up to the last 20%



#%%

# train_data = train_df[RATIO_TO_PREDICT].values
# valid_data = validation_df[RATIO_TO_PREDICT].values
# test_data = test_df[RATIO_TO_PREDICT].values
# train_data = train_data.reshape(-1,1)
# valid_data = valid_data.reshape(-1,1)
# test_data = test_data.reshape(-1,1)

train_data = train_df[RATIO_TO_PREDICT].values
valid_data = validation_df[RATIO_TO_PREDICT].values
test_data = test_df[RATIO_TO_PREDICT].values


X_train = []
y_train = []
for i in range(SEQ_LEN, len(train_data)):
    X_train.append(train_data[i-SEQ_LEN:i])
    y_train.append(train_data[i + (FUTURE_PERIOD_PREDICT-1)])
X_train, y_train = np.array(X_train), np.array(y_train)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_valid = []
y_valid = []
for i in range(SEQ_LEN, len(valid_data)):
    X_valid.append(valid_data[i-SEQ_LEN:i])
    y_valid.append(valid_data[i+(FUTURE_PERIOD_PREDICT-1)])
X_valid, y_valid = np.array(X_valid), np.array(y_valid)
# X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

X_test = []
y_test = []
for i in range(SEQ_LEN, len(test_data)):
    X_test.append(test_data[i-SEQ_LEN:i])
    y_test.append(test_data[i+(FUTURE_PERIOD_PREDICT-1)])
X_test, y_test = np.array(X_test), np.array(y_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

#%%
import tensorflow as tf
from forex.NN_Models import NN_Models
from sklearn.utils import shuffle

weight_filename = 'lstm-bert-BATCH_SIZE_'+str(BATCH_SIZE)+'-SEQ_LEN_'+str(SEQ_LEN)+'-FUTURE_PERIOD_PREDICT_'+str(FUTURE_PERIOD_PREDICT)+'.h5'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

nn_models = NN_Models()
lstm_bert = nn_models.get_LSTM_BERT(SEQ_LEN)

#%%
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
lstm_bert.save_weights(path.join(dataDir, 'USDJPY', 'lstm-bert.h5'))

#%%
print('X_test shape: ')
print(X_test.shape)
predicted_stock_price_multi_head = lstm_bert.predict(X_test)
print('predicted_stock_price_multi_head shape: ')
print(predicted_stock_price_multi_head.shape)
predicted_stock_price_multi_head = np.vstack((np.full((SEQ_LEN,1), np.nan), predicted_stock_price_multi_head))

session.close()

#%%
reverseScaled_predicted_stock_price_multi_head = np.copy(predicted_stock_price_multi_head)

actualVals = test_df[['Close']].values
predictedVals = []
for i in range(SEQ_LEN, len(reverseScaled_predicted_stock_price_multi_head)):
    minMaxScaler.fit(actualVals[i-SEQ_LEN:i])
    reverseScaled_predicted_stock_price_multi_head[i] = minMaxScaler.inverse_transform([reverseScaled_predicted_stock_price_multi_head[i]])[0]

reverseScaled_predicted_stock_price_multi_head

#%%
import matplotlib.pyplot as plt
# print(predicted_stock_price_multi_head)
# plt.figure(figsize = (18,9))
# plt.plot(test_df[['Close']].values, color = 'red', label = 'Actual Price')
# plt.plot(reverseScaled_predicted_stock_price_multi_head, color = 'green', label = 'Predicted Price')
# plt.title('USDJPY Price Prediction', fontsize=30)
# #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend(fontsize=18)
# plt.show()

#%%
prices = test_df[['Close']].values
# act_diff = np.diff(prices.reshape(1,-1)[0])
predicted_diff = np.diff(reverseScaled_predicted_stock_price_multi_head.reshape(1,-1)[0]) #predicted_stock_price_multi_head.reshape(1, -1)[0] #np.diff(reverseScaled_predicted_stock_price_multi_head.reshape(1,-1)[0])

print('predicted_diff shape:')
print(predicted_diff.shape)

long_cash_balance = 0
long_holding = 0

short_cash_balance = 0
short_holding = 0

records = [0]
balance = [0]
for i in range(0, len(predicted_diff)):
    if not np.isfinite(predicted_diff[i]):
        records.append(0)
        balance.append(0)
        continue
    if i>0 and not np.isfinite(predicted_diff[i-1]):
        records.append(0)
        balance.append(0)
        continue

    curPrice = prices[i-1][0]
    if predicted_diff[i] > predicted_diff[i-1]:
        long_cash_balance -= curPrice
        long_holding += 1

        short_cash_balance -= (short_holding * curPrice)
        short_holding = 0

        records.append(1)
    elif predicted_diff[i] < predicted_diff[i-1]:
        long_cash_balance += (long_holding * curPrice)
        long_holding = 0

        short_cash_balance += curPrice
        short_holding += 1

        records.append(-1)
    else:
        records.append(0)
    balance.append((long_cash_balance+(long_holding * curPrice)) + (short_cash_balance-(short_holding * curPrice)))

print('Long Holding & Balance:')
print(long_holding)
print(long_cash_balance)

print('Short Holding & Balance:')
print(short_holding)
print(short_cash_balance)

print(balance[-1])

# plt.figure(figsize = (18,9))
# plt.plot(balance, color = 'red', label = 'Actual Price')
# plt.plot(balance, color = 'red', label = 'Actual Price')
# plt.title('USDJPY Price Prediction', fontsize=30)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend(fontsize=18)
# plt.show()


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Price', color=color)
ax1.plot(prices, label='Price', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(balance, label='Balance')
ax2.legend()

fig.tight_layout()
plt.show()










# #%%

# p = OHLC(path.join(dataDir, 'USDJPY_1MIN_(1-1-2008_31-12-2017).csv'))
# # p.set_df(p.get_df_with_resolution('30min'))

# p.df.reset_index(drop=True, inplace=True)

# print(p.df['Close'])
# minMaxScale = lambda x: minMaxScaler.fit_transform(x.reshape(-1,1)).reshape(1, len(x))[0][-1]
# rollingWindow = 24 * 30 * 6
# p.df['MinMaxScaled_Open'] = p.df['Open'].rolling(rollingWindow).apply(minMaxScale)
# p.df['MinMaxScaled_High'] = p.df['High'].rolling(rollingWindow).apply(minMaxScale)
# p.df['MinMaxScaled_Low'] = p.df['Low'].rolling(rollingWindow).apply(minMaxScale)
# p.df['MinMaxScaled_Close'] = p.df['Close'].rolling(rollingWindow).apply(minMaxScale)

# #%%
# df = p.df

# #%%
# allData = df[['MinMaxScaled_Close']].values
# X_allData = []
# for i in range(SEQ_LEN, len(allData)):
#     X_allData.append(allData[i-SEQ_LEN:i])

# predicted_stock_price_multi_head_val = lstm_bert.predict(np.array(X_allData))
# print(predicted_stock_price_multi_head_val.shape)
# predicted_stock_price_multi_head_val = np.vstack((np.full((SEQ_LEN,1), np.nan), predicted_stock_price_multi_head_val))

# reverseScaled_predicted_stock_price_multi_head_val = np.copy(predicted_stock_price_multi_head_val)

# actualVals = df[['Close']].values
# predictedVals = []
# for i in range(SEQ_LEN, len(reverseScaled_predicted_stock_price_multi_head_val)):
#     minMaxScaler.fit(actualVals[i-SEQ_LEN:i])
#     reverseScaled_predicted_stock_price_multi_head_val[i] = minMaxScaler.inverse_transform([reverseScaled_predicted_stock_price_multi_head_val[i]])[0]

# reverseScaled_predicted_stock_price_multi_head_val


# act_diff = np.diff(df[['Close']].values.reshape(1,-1)[0])
# predicted_diff = np.diff(reverseScaled_predicted_stock_price_multi_head_val.reshape(1,-1)[0])

# prices = df[['Close']].values
# cash_balance = 0
# holding = 0

# records = [0]
# balance = [0]
# for i in range(0, len(predicted_diff)):
#     if not np.isfinite(predicted_diff[i]):
#         records.append(0)
#         balance.append(0)
#         continue
#     if i>0 and not np.isfinite(predicted_diff[i-1]):
#         records.append(0)
#         balance.append(0)
#         continue

#     curPrice = prices[i-1][0]
#     if predicted_diff[i] > predicted_diff[i-1]:
#         cash_balance -= curPrice
#         holding += 1
#         records.append(1)
#     elif predicted_diff[i] < predicted_diff[i-1]:
#         cash_balance += curPrice
#         holding -= 1
#         records.append(-1)
#     else:
#         records.append(0)
#     balance.append(cash_balance+(holding * curPrice))

# print(holding)
# print(cash_balance)
# plt.plot(balance, color = 'red', label = 'Actual Price')


