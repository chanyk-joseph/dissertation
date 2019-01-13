import os.path as path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from forex.utils import *
from strategies.sma import SMA
from strategies.macd import MACD

import talib

def plot(df):
    df.plot(x='datetime', y='total_asset_value')
    plt.show()

def process_row(df, rowIndex):
    df.set_value(rowIndex, 'x', 1111)
    print(df)

script_dir = path.dirname(path.realpath(sys.argv[0]))
data_dir = "F:\\modified_data"

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
df = df.resample('360min').pad()
macd = MACD()
df = macd.generate_buy_sell_records(df)

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