import os.path as path
import sys

import pandas as pd
# import matplotlib.pyplot as plt

from forex.utils import *
from strategies.sma import SMA

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

df = set_df_Timestamp_as_datetime(pd.read_csv(ohlcCSV))
# print(get_df_by_datetime_range(df, parseISODateTime('2008-01-02T00:00:00'), parseISODateTime('2017-01-03T00:00:00')))
# close = np.random.random(100)
smaStrategy = SMA(60*60*24*3)
result = smaStrategy.calculate(df, parseISODateTime('2001-01-07T00:00:00'))
print(result)
# plt.plot(result)
# plt.ylabel('some numbers')
# plt.show()