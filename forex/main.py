import os.path as path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from forex.utils import *
from strategies.sma import SMA
from strategies.macd import MACD

from sklearn.preprocessing import MinMaxScaler
from data import OHLC

import talib

script_dir = path.dirname(path.realpath(sys.argv[0]))
data_dir = "F:\\modified_data"

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
for currencyPair in currencyPairs:
    p = OHLC(path.join('F:\\modified_data', currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
    p.set_df(p.get_df_with_resolution('1min'))
    p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
    p.merge_df(p.get_normalized_price([60, 120, 240, 1440, 10080, 43200], 'mins'))
    p.save(path.join('F:\\modified_data', currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_returns.csv'))

sys.exit()

p.df['Close'] = p.df['Close']
print(p.df.tail(400).loc[:, ['Close', 'Close_Normalized_By_Future_ema360', 'Close_Normalized_By_Future_360']])
# p.print()
plot_fields(get_df_by_datetime_range(p.df, parseISODateTime('2008-01-02T00:00:00'), parseISODateTime('2008-01-03T00:00:00')), 'Close', ['Close_Normalized_By_Past_1800', 'Close_Normalized_By_Future_1800'])

# print(p.get_df_with_resolution('2D'))
# print(pd.infer_freq(p.df.index))
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