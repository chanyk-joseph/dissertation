import sys
import platform
import os.path as path
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pandas as pd

from forex.OHLC import OHLC

dataDir = ''
if platform.system() == 'Windows':
    dataDir = 'F:\\modified_data'
elif platform.system() == 'Linux':
    dataDir = '/home/joseph/Desktop/datastore'
else:
    exit()

currencyPairs = ['AUDUSD', 'EURGBP', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDJPY', 'USDCAD', 'USDCHF', 'XAUUSD']

######################################## MultiThreading ########################################
# def append_return(currencyPair):
#     p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
#     p.set_df(p.get_df_with_resolution('1min'))
#     p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
#     p.merge_df(p.get_normalized_price([15, 30, 60, 120, 240, 1440, 10080, 43200], 'mins'))
#     p.save(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_returns_3.csv'))
#     return currencyPair
# def main():
#     executor = ThreadPoolExecutor(3)
#     futures = []
#     for currencyPair in currencyPairs:
#         futures.append(executor.submit(append_return, (currencyPair)))

#     while(1):
#         allDone = all(f.done() == True for f in futures)
#         if allDone:
#             break
#         sleep(1)
#     exit()
# if __name__ == '__main__':
#     main()

######################################## Append returns Columns ########################################
# currencyPairs = ['USDJPY']
# for currencyPair in currencyPairs:
#     p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
#     p.set_df(p.get_df_with_resolution('1min'))
#     p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
#     p.merge_df(p.get_normalized_price([15, 30, 60, 120, 240, 1440, 10080, 43200], 'mins'))
#     p.save(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_returns.csv'))
# exit()
# ['Timestamp' 'Open' 'High' 'Low' 'Close' 'Volume' 'LogReturn_1mins'
#  'LogReturn_forward_1mins_sum' 'LogReturn_forward_2mins_sum'
#  'LogReturn_forward_4mins_sum' 'LogReturn_forward_8mins_sum'
#  'LogReturn_forward_16mins_sum' 'LogReturn_forward_32mins_sum'
#  'LogReturn_forward_1mins_min' 'LogReturn_forward_1mins_max'
#  'LogReturn_forward_2mins_min' 'LogReturn_forward_2mins_max'
#  'LogReturn_forward_4mins_min' 'LogReturn_forward_4mins_max'
#  'LogReturn_forward_8mins_min' 'LogReturn_forward_8mins_max'
#  'LogReturn_forward_16mins_min' 'LogReturn_forward_16mins_max'
#  'LogReturn_forward_32mins_min' 'LogReturn_forward_32mins_max'
#  'PIP_Return_1mins' 'PIP_Return_forward_looking_for_1mins_sum'
#  'PIP_Return_forward_looking_for_2mins_sum'
#  'PIP_Return_forward_looking_for_4mins_sum'
#  'PIP_Return_forward_looking_for_8mins_sum'
#  'PIP_Return_forward_looking_for_16mins_sum'
#  'PIP_Return_forward_looking_for_32mins_sum'
#  'PIP_Return_forward_looking_for_1mins_min'
#  'PIP_Return_forward_looking_for_1mins_max'
#  'Normalized_PIP_Return_forward_looking_for_1mins_max'
#  'PIP_Return_forward_looking_for_2mins_min'
#  'PIP_Return_forward_looking_for_2mins_max'
#  'Normalized_PIP_Return_forward_looking_for_2mins_max'
#  'PIP_Return_forward_looking_for_4mins_min'
#  'PIP_Return_forward_looking_for_4mins_max'
#  'Normalized_PIP_Return_forward_looking_for_4mins_max'
#  'PIP_Return_forward_looking_for_8mins_min'
#  'PIP_Return_forward_looking_for_8mins_max'
#  'Normalized_PIP_Return_forward_looking_for_8mins_max'
#  'PIP_Return_forward_looking_for_16mins_min'
#  'PIP_Return_forward_looking_for_16mins_max'
#  'Normalized_PIP_Return_forward_looking_for_16mins_max'
#  'PIP_Return_forward_looking_for_32mins_min'
#  'PIP_Return_forward_looking_for_32mins_max'
#  'Normalized_PIP_Return_forward_looking_for_32mins_max'
#  'is_all_forward_looking_increasing_within_1mins'
#  'is_all_forward_looking_increasing_within_2mins'
#  'is_all_forward_looking_increasing_within_4mins'
#  'is_all_forward_looking_increasing_within_8mins'
#  'is_all_forward_looking_increasing_within_16mins'
#  'is_all_forward_looking_increasing_within_32mins'
#  'Close_Normalized_By_Past_15mins' 'Close_Normalized_By_Future_15mins'
#  'Close_Normalized_By_Past_30mins' 'Close_Normalized_By_Future_30mins'
#  'Close_Normalized_By_Past_60mins' 'Close_Normalized_By_Future_60mins'
#  'Close_Normalized_By_Past_120mins' 'Close_Normalized_By_Future_120mins'
#  'Close_Normalized_By_Past_240mins' 'Close_Normalized_By_Future_240mins'
#  'Close_Normalized_By_Past_1440mins' 'Close_Normalized_By_Future_1440mins'
#  'Close_Normalized_By_Past_10080mins'
#  'Close_Normalized_By_Future_10080mins'
#  'Close_Normalized_By_Past_43200mins'
#  'Close_Normalized_By_Future_43200mins']

######################################## Extract subset of rows ########################################
# currencyPair = 'USDJPY'
# p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_returns.csv'))
# df = p.df
# times = sorted(df.index.values)
# last_10pct = sorted(df.index.values)[-int(0.1*len(times))]  # get the last 10% of the times
# last_12pct = sorted(df.index.values)[-int(0.12*len(times))]  # get the last 12% of the times
# p.df = df[(df.index >= last_12pct) & (df.index < last_10pct)]
# p.print()
# p.save(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_returns_simple.csv'))



currencyPair = 'USDJPY'
# p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_returns_simple.csv'))
p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
p.set_df(p.get_df_with_resolution('1min'))
p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
# p.merge_df(p.get_normalized_price([15, 30, 60, 120, 240, 1440, 10080, 43200], 'mins'))
print(len(p.df.index))


# arrKeys = ['Normalized_PIP_Return_forward_looking_for_1mins_max',
    # 'Normalized_PIP_Return_forward_looking_for_2mins_max',
    # 'Normalized_PIP_Return_forward_looking_for_4mins_max',
    # 'Normalized_PIP_Return_forward_looking_for_8mins_max',
    # 'Normalized_PIP_Return_forward_looking_for_16mins_max',
    # 'Normalized_PIP_Return_forward_looking_for_32mins_max'
arrKeys = ['PIP_Return_forward_looking_for_1mins_min',
    'PIP_Return_forward_looking_for_1mins_max',
    'PIP_Return_forward_looking_for_2mins_min',
    'PIP_Return_forward_looking_for_2mins_max',
    'PIP_Return_forward_looking_for_4mins_min',
    'PIP_Return_forward_looking_for_4mins_max',
    'PIP_Return_forward_looking_for_8mins_min',
    'PIP_Return_forward_looking_for_8mins_max',
    'PIP_Return_forward_looking_for_16mins_min',
    'PIP_Return_forward_looking_for_16mins_max',
    'PIP_Return_forward_looking_for_32mins_min',
    'PIP_Return_forward_looking_for_32mins_max',
]

results = {}
cols = []
summary = p.df.loc[:, arrKeys].describe()
for resol in [1,2,4,8,16,32]:
    resolStr = 'PIP_Return_forward_looking_for_' + str(resol) + 'mins'
    for stat in ['mean', 'std', '75%']:
        colName = str(resol)+'mins_' + stat + '_returns'
        cols.append(colName)
        results[colName] = summary[resolStr+'_max'][stat]

newdf  = pd.DataFrame(columns = cols)
newdf.loc[0] = results
print(newdf)

# print(results)

# print(p.df.loc[:, arrKeys].describe()['PIP_Return_forward_looking_for_1mins_min']['count'])