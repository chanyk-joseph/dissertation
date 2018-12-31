import os.path as path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import talib
from forex.utils import *


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

df = pd.read_csv(ohlcCSV)
df = set_df_Timestamp_as_datetime(df)
print(get_df_by_datetime_range(df, parseISODateTime('2008-01-02T00:00:00'), parseISODateTime('2017-01-03T00:00:00')))




# import talib
# close = np.random.random(100)
# SMA = talib.MA(close,30,matype=0)

# plt.plot(SMA)
# plt.ylabel('some numbers')
# plt.show()