
# import random, os, sys
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import *
# from tensorflow.keras.initializers import *
# import tensorflow.keras.backend as K
# import tensorflow as tf
# from tensorflow.python.keras.layers import Layer
# from tensorflow.python.keras.layers import CuDNNLSTM
# LSTM = CuDNNLSTM


import tensorflow
print(tensorflow.__version__)

import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

print(keras.__version__)

import tensorflow.python.keras.layers.CuDNNLSTM as LSTM
import numpy

# The inputs to the model.
# We will create two data points, just for the example.
data_x = numpy.array([
    # Datapoint 1
    [
        # Input features at timestep 1
        [1, 2, 3],
        # Input features at timestep 2
        [4, 5, 6]
    ],
    # Datapoint 2
    [
        # Features at timestep 1
        [7, 8, 9],
        # Features at timestep 2
        [10, 11, 12]
    ]
])

# The desired model outputs.
# We will create two data points, just for the example.
data_y = numpy.array([
    # Datapoint 1
    [
        # Target features at timestep 1
        [101, 102, 103, 104],
        # Target features at timestep 2
        [105, 106, 107, 108]
    ],
    # Datapoint 2
    [
        # Target features at timestep 1
        [201, 202, 203, 204],
        # Target features at timestep 2
        [205, 206, 207, 208]
    ]
])

# Each input data point has 2 timesteps, each with 3 features.
# So the input shape (excluding batch_size) is (2, 3), which
# matches the shape of each data point in data_x above.
model_input = L.Input(shape=(2, 3))

# This RNN will return timesteps with 4 features each.
# Because return_sequences=True, it will output 2 timesteps, each
# with 4 features. So the output shape (excluding batch size) is
# (2, 4), which matches the shape of each data point in data_y above.
model_output = LSTM(4, return_sequences=True)(model_input)

print('----')
print(model_input)
print('----')
model_output
print('----')

# Create the model.
model = M.Model(input=model_input, output=model_output)

# You need to pick appropriate loss/optimizers for your problem.
# I'm just using these to make the example compile.
model.compile('sgd', 'mean_squared_error')

# Train
model.fit(data_x, data_y)

sys.exit()

import os.path as path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from sklearn.preprocessing import MinMaxScaler
from OHLC import OHLC

script_dir = path.dirname(path.realpath(sys.argv[0]))
data_dir = "/hdd/dissertation"

def process_row(df, rowIndex):
    df.set_value(rowIndex, 'x', 1111)
    print(df)

def plot_earn_graph(df, buyColName):
    positions = []
    balance = 0
    df.at[list(df.index)[0], "balance"] = 0
    close_i = df.columns.get_loc('Close') 
    buyNext = False
    for i, row in enumerate(df.itertuples()):
        curTimestamp = row.Timestamp.to_pydatetime()
        curClose = row.Close
        isIncreasing = getattr(row, buyColName)

        if buyNext:
            positions.append({
                'Close': curClose,
                'Timestamp': row.Timestamp.to_pydatetime()
            })
            buyNext = False
        if isIncreasing == 1:
            buyNext = True
        if len(positions) > 0:
            shouldCutPositions = [pos for pos in positions if curClose - pos['Close'] <= -0.0050 or curClose - pos['Close'] >= 0.02 or (curTimestamp - pos['Timestamp']).total_seconds() >= 1800]
            for pendingCutPos in shouldCutPositions:
                netProfit = curClose - pendingCutPos['Close']
                # print(str(i) + ": " + str(netProfit) + " | balance: "+ str(balance))
                balance += netProfit * 10000
                positions.remove(pendingCutPos)
        df.loc[row.Index, 'balance'] = balance
    plot_fields(df, ['balance'])

currencyPairs = ['AUDUSD', 'EURGBP', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDJPY', 'USDCAD', 'USDCHF', 'XAUUSD']


import random, os, sys
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import CuDNNLSTM
LSTM = CuDNNLSTM

try:
    from dataloader import TokenList, pad_to_longest
    # for transformer
except: pass



embed_size = 60

SEQ_LEN = 200  # how long of a preceeding sequence to collect for RNN

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def __call__(self, x):
        output = self.w_1(x) 
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
        if pos != 0 else np.zeros(d_emb) 
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class Transformer():
    def __init__(self, len_limit, embedding_matrix, d_model=embed_size, \
              d_inner_hid=512, n_head=10, d_k=64, d_v=64, layers=2, dropout=0.1, \
              share_word_emb=False, **kwargs):
        self.name = 'Transformer'
        self.len_limit = len_limit
        self.src_loc_info = False # True # sl: fix later
        self.d_model = d_model
        self.decode_model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                            weights=[GetPosEncodingMatrix(len_limit, d_emb)])

        i_word_emb = Embedding(max_features, d_emb, weights=[embedding_matrix]) # Add Kaggle provided embedding here

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                               word_emb=i_word_emb, pos_emb=pos_emb)

        
    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile(self, active_layers=999):
        src_seq_input = Input(shape=(None, ))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(src_seq_input)
        
        # LSTM before attention layers
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x) 
        
        x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
        
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        x = Dense(1, activation="sigmoid")(conc)   
        
        
        self.model = Model(inputs=src_seq_input, outputs=x)
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

def build_model():
    inp = Input(shape = (SEQ_LEN, 1))
    
    # LSTM before attention layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x) 
        
    x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
        
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    x = Dense(1, activation="sigmoid")(conc)      

    model = Model(inputs = inp, outputs = x)
    model.compile(
        loss = "mean_squared_error", 
        #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 
        optimizer = "adam")
    
    # Save entire model to a HDF5 file
    #model.save('my_model.h5')
    
    return model

multi_head = build_model()
multi_head.summary()

RATIO_TO_PREDICT = 'Normalized_PIP_Return_forward_looking_for_8mins_max'
FUTURE_PERIOD_PREDICT = 1

print('Reading CSV')
p = OHLC(path.join('/hdd/dissertation', 'USDJPY_1MIN_(1-1-2008_31-12-2017)_with_returns.csv'))
print('Readed')
p.set_df(p.get_df_with_resolution('8min'))
p.df.reset_index(drop=True)
df = p.df.loc[:, [RATIO_TO_PREDICT]]
# p.save(path.join('/hdd/dissertation', 'USDJPY_1MIN_(1-1-2008_31-12-2017)_with_returns_2.csv'))
# df = pd.read_csv(path.join('/hdd/dissertation', 'USDJPY_1MIN_(1-1-2008_31-12-2017)_with_returns_2.csv'),delimiter=',',usecols=[RATIO_TO_PREDICT])
df = df.dropna()

# df = pd.read_csv(path.join('/hdd/dissertation', 'USDJPY_1MIN_(1-1-2008_31-12-2017)_with_returns.csv'),delimiter=',',usecols=[RATIO_TO_PREDICT])
# df = df.dropna()

times = sorted(df.index.values)  # get the times
last_10pct = sorted(df.index.values)[-int(0.1*len(times))]  # get the last 10% of the times
last_20pct = sorted(df.index.values)[-int(0.2*len(times))]  # get the last 20% of the times

test_df = df[(df.index >= last_10pct)]
validation_df = df[(df.index >= last_20pct) & (df.index < last_10pct)]  
train_df = df[(df.index < last_20pct)]  # now the train_df is all the data up to the last 20%

train_data = train_df[RATIO_TO_PREDICT].as_matrix()
valid_data = validation_df[RATIO_TO_PREDICT].as_matrix()
test_data = test_df[RATIO_TO_PREDICT].as_matrix()
train_data = train_data.reshape(-1,1)
valid_data = valid_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

X_train = []
y_train = []
for i in range(SEQ_LEN, len(train_data)):
    X_train.append(train_data[i-SEQ_LEN:i])
    y_train.append(train_data[i + (FUTURE_PERIOD_PREDICT-1)])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_valid = []
y_valid = []
for i in range(SEQ_LEN, len(valid_data)):
    X_valid.append(valid_data[i-SEQ_LEN:i])
    y_valid.append(valid_data[i+(FUTURE_PERIOD_PREDICT-1)])
X_valid, y_valid = np.array(X_valid), np.array(y_valid)
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

X_test = []
y_test = []
for i in range(SEQ_LEN, len(test_data)):
    X_test.append(test_data[i-SEQ_LEN:i])
    y_test.append(test_data[i+(FUTURE_PERIOD_PREDICT-1)])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 1024  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
import time
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model

multi_head.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_valid, y_valid), 
                    #callbacks = [checkpoint , lr_reduce]
             )

predicted_stock_price_multi_head = multi_head.predict(X_test)
print(predicted_stock_price_multi_head.shape)
predicted_stock_price_multi_head = np.vstack((np.full((60,1), np.nan), predicted_stock_price_multi_head))


plt.figure(figsize = (18,9))
# plt.plot(test_data, color = 'black', label = 'GE Stock Price')
plt.plot(predicted_stock_price_multi_head, color = 'green', label = 'Predicted GE Mid Price')
plt.title('GE Mid Price Prediction', fontsize=30)
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('GE Mid Price')
plt.legend(fontsize=18)
plt.show()

print('sss')
sys.exit()


# p = OHLC(path.join('F:\\modified_data', 'USDJPY_Daily_(1-1-2008_31-12-2017).csv'))
# p.plot_fields('Close')
# sys.exit()

# p = OHLC(path.join('F:\\modified_data', 'balance_forward_16_04.csv'))
# p.plot_fields('Balance')
# sys.exit()

p = OHLC(path.join('F:\\modified_data', 'USDJPY_1MIN_(1-1-2008_31-12-2017)_with_returns.csv'))
p.plot_fields('Normalized_PIP_Return_forward_looking_for_16mins_max', [], parseISODateTime('2010-01-01T00:00:00'), parseISODateTime('2010-12-31T00:00:00'))
sys.exit()

# p = OHLC(path.join('F:\\modified_data', 'USDJPY_1MIN_(1-1-2008_31-12-2017).csv'))
# p.set_df(p.get_df_with_resolution('1min'))
# p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
# p.merge_df(p.get_normalized_price([15, 30, 60, 120, 240, 1440, 10080, 43200], 'mins'))
# p.save(path.join('F:\\modified_data', 'USDJPY_1MIN_(1-1-2008_31-12-2017)_with_returns_2.csv'))
# sys.exit()

for currencyPair in currencyPairs:
    p = OHLC(path.join('F:\\modified_data', currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
    p.set_df(p.get_df_with_resolution('1min'))
    p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
    p.merge_df(p.get_normalized_price([15, 30, 60, 120, 240, 1440, 10080, 43200], 'mins'))
    p.save(path.join('F:\\modified_data', currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_returns.csv'))
sys.exit()


cl = df['Close'].as_matrix().reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(cl)
cl = scaler.transform(cl)
print(cl)
df['norm_Close'] = cl
print(df)
df['Close'] = df['Close'] / 100
plot_fields(df, ['Close', 'norm_Close'])

'''
tradeRecordCSV = path.join(script_dir, 'trade-record-simple.csv')
ohlcCSV = path.join(data_dir, 'EURUSD_Daily_(1-1-2008_31-12-2017).csv')
# df = set_df_Timestamp_as_datetime(pd.read_csv(ohlcCSV))
# print(get_df_by_datetime_range(df, parseISODateTime('2008-01-02T00:00:00'), parseISODateTime('2017-01-03T00:00:00')))
# close = np.random.random(100)
# smaStrategy = SMA(60*60*24*3) 
# result = smaStrategy.calculate(df, parseISODateTime('2001-01-07T00:00:00'))
# print(result)
# plt.plot(result)
# plt.ylabel('some numbers')
# plt.show()


# ohlc_s_csv = path.join(data_dir, 'EURUSD_1MIN_(1-1-2008_31-12-2017).csv')
# df = set_df_Timestamp_as_datetime(pd.read_csv(ohlc_s_csv))
# df.index = pd.DatetimeIndex(df['Timestamp'])
# df = df.resample('1D').pad()
# macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
# plt.plot(macdhist, label='macdhist')
# macd = MACD()
# singal, signal_strength = macd.generate_signal(df)
# plt.plot(signal_strength / 50, label='signal_strength')
# plt.plot(singal * signal_strength / 50, label='signal')
# plt.axhline(y=0, color='r', linestyle='-')
# plt.ylabel('some numbers')
# plt.legend()
# plt.show()

ohlc_s_csv = path.join(data_dir, 'USDJPY_1MIN_(1-1-2008_31-12-2017).csv')
df = set_df_Timestamp_as_datetime(pd.read_csv(ohlc_s_csv))
df.index = pd.DatetimeIndex(df['Timestamp'])
df = df.resample('5min').pad() # pad: fill NaN with previous value 




# Calculate log return, and forward looking to generate
df['LogReturn'] = np.log(df.Close/df.Close.shift(1))
df['Forward5LogReturn'] = df['LogReturn'][::-1].rolling(5).sum()[::-1] - df['LogReturn']
df['Forward10LogReturn'] = df['LogReturn'][::-1].rolling(10).sum()[::-1] - df['LogReturn']
# df['Forward15LogReturn'] = df['LogReturn'][::-1].rolling(15).sum()[::-1] - df['LogReturn']
# df['Forward30LogReturn'] = df['LogReturn'][::-1].rolling(30).sum()[::-1] - df['LogReturn']
# df['Forward60LogReturn'] = df['LogReturn'][::-1].rolling(60).sum()[::-1] - df['LogReturn']
# df['Forward360LogReturn'] = df['LogReturn'][::-1].rolling(360).sum()[::-1] - df['LogReturn']

print(df.head(10))
print(df.tail(10))

records = df
# plt.plot(records["Close"] / 100000, label='Close')
plt.plot(records["Forward5LogReturn"], label='Forward5LogReturn')
plt.plot(records["Forward10LogReturn"], label='Forward10LogReturn')
# plt.plot(records["Forward5LogReturn"], label='Forward5LogReturn')
# plt.plot(records["Forward15LogReturn"], label='Forward15LogReturn')
# plt.plot(records["Forward30LogReturn"], label='Forward30LogReturn')
# plt.plot(records["Forward60LogReturn"], label='Forward60LogReturn')
# plt.plot(records["Forward360LogReturn"], label='Forward360LogReturn')
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('some numbers')
plt.legend()
plt.show()

sys.exit()





macd = MACD()
df = macd.generate_buy_sell_records(df)




print(df.head())
df = df.loc[abs(df["MACD"])==1]
print(df.head())
print(len(df.index))


records = df.loc[abs(df['MACD'])==1]
records['CloseDiff'] = records['Close'].diff()
records['MACDDiff'] = records['MACD'].diff()
records.at[list(records.index)[0], "cash_balance"] = 100
records.at[list(records.index)[0], "holding_units"] = 0
records.at[list(records.index)[0], "total_asset_value"] = 100
print(records.head())

close_i = records.columns.get_loc('Close')
close_diff_i = records.columns.get_loc('CloseDiff')
macd_col_i = records.columns.get_loc('MACD')
macd_diff_col_i = records.columns.get_loc('MACDDiff')
cash_balance_i = records.columns.get_loc('cash_balance')
holding_units_i = records.columns.get_loc('holding_units')
total_asset_value_i = records.columns.get_loc('total_asset_value')
for i, row in enumerate(records.itertuples()):
    if i == 0:
        continue

    curCloseDiff = records.iat[i, close_diff_i]
    curMACD = records.iat[i, macd_col_i]
    curMACDDiff = records.iat[i, macd_diff_col_i]

    if curMACDDiff == -2:
        records.iat[i, total_asset_value_i] = records.iat[i-1, total_asset_value_i] + curCloseDiff * 100
    else:
        records.iat[i, total_asset_value_i] = records.iat[i-1, total_asset_value_i]

print(records.tail())
plt.plot(records["Close"] * 100, label='Close')
plt.plot(records["total_asset_value"], label='total_asset_value')
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('some numbers')
plt.legend()
plt.show()

sys.exit()





df.at[list(df.index)[0], "cash_balance"] = 100
df.at[list(df.index)[0], "holding_units"] = 0
df.at[list(df.index)[0], "total_asset_value"] = 100

print(df.head())
for i, row in enumerate(df.itertuples()):
    if i == 0:
        continue

    close_i = df.columns.get_loc('Close')
    macd_col_i = df.columns.get_loc('MACD')
    cash_balance_i = df.columns.get_loc('cash_balance')
    holding_units_i = df.columns.get_loc('holding_units')
    total_asset_value_i = df.columns.get_loc('total_asset_value')

    curClose = df.iat[i, close_i]
    curMACD = df.iat[i, macd_col_i]
    curCash = df.iat[i, cash_balance_i] = df.iat[i-1, cash_balance_i]
    curHolding = df.iat[i, holding_units_i] = df.iat[i-1, holding_units_i]
    curTotal = df.iat[i, total_asset_value_i] = df.iat[i-1, total_asset_value_i]

    if curMACD == 1 and curCash > 0:
        curHolding += curCash / curClose
        curCash = 0
    elif curMACD == -1 and curHolding > 0:
        curCash += curHolding * curClose
        curHolding = 0
    curTotal = curHolding * curClose + curCash

    df.iat[i, cash_balance_i] = curCash
    df.iat[i, holding_units_i] = curHolding
    df.iat[i, total_asset_value_i] = curTotal
    
print(df.tail())
plt.plot(df["Close"] * 100, label='Close')
plt.plot(df["total_asset_value"], label='total_asset_value')
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('some numbers')
plt.legend()
plt.show()
'''