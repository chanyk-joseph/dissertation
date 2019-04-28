
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

#%% Read data
import os.path as path
import pandas as pd
from forex.OHLC import OHLC

currencyPair = 'USDJPY'
p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
p.df.drop(['Timestamp'], axis=1)

#%% Convert to 1min interval and fill missing records with previous values
p.set_df(p.get_df_with_resolution('1min'))

#%% Calculate pip returns within future X mins
p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
p.df.dtypes

#%% 
p.df.drop(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

#%% 
pd.DataFrame({
    '1mins': p.df['PIP_Return_forward_looking_for_1mins_max'],
    '2mins': p.df['PIP_Return_forward_looking_for_2mins_max'],
    '4mins': p.df['PIP_Return_forward_looking_for_4mins_max'],
    '8mins': p.df['PIP_Return_forward_looking_for_8mins_max'],
    '16mins': p.df['PIP_Return_forward_looking_for_16mins_max'],
    '32mins': p.df['PIP_Return_forward_looking_for_32mins_max']
})

#%% Select subset of data
# from forex.utils import *
# mask = (p.df['Timestamp'] >= parseISODateTime('2011-05-01T00:00:00')) & (p.df['Timestamp'] <= parseISODateTime('2011-08-12T00:00:00'))
# p.set_df(p.df.loc[mask])

#%% Maximum actual returns within following mins in the future
import numpy as np
key = 'max'
arrKeys = ['PIP_Return_forward_looking_for_1mins_'+key,
    'PIP_Return_forward_looking_for_2mins_'+key,
    'PIP_Return_forward_looking_for_4mins_'+key,
    'PIP_Return_forward_looking_for_8mins_'+key,
    'PIP_Return_forward_looking_for_16mins_'+key,
    'PIP_Return_forward_looking_for_32mins_'+key
]

indexes = []
results = {'mean':[], 'std':[], '75%':[], '95%':[]}
summary = p.df.loc[:, arrKeys].describe()
for resol in [1,2,4,8,16,32]:
    indexes.append(str(resol)+'mins')
    resolStr = 'PIP_Return_forward_looking_for_' + str(resol) + 'mins'
    for stat in ['mean', 'std', '75%']:
        results[stat].append(summary[resolStr+'_'+key][stat])
    results['95%'].append(np.percentile(p.df[resolStr+'_'+key].dropna(), 95))
newdf = pd.DataFrame(results, index=indexes)
newdf

#%% Get the percentile boundary
import numpy as np
percentiles = ['3', '16', '50', '84', '97']
results = {}
colToBeUsed = p.df['PIP_Return_forward_looking_for_32mins_max'].dropna()
for stat in percentiles:
    results[stat] = np.percentile(colToBeUsed, int(stat))
results


#%% Convert percentile into signal classes: [
# 1 (0%  <= x <  3%), 
# 2 (3%  <= x <  16%), 
# 3 (16% <= x <  50%), 
# 4 (50% <= x <  84%), 
# 5 (84% <= x <  97%), 
# 6 (97% <= x <= 100%)
# ]

from IPython.core.display import Image, display
display(Image(url="https://timotheories.files.wordpress.com/2017/01/standard_deviation_diagram-svg.png?w=1000&h=700&crop=1", width=800, height=500))

import pandas as pd
bins = [-9999999]
for percentile in percentiles:
    bins.append(results[percentile])
bins.append(9999999)
# bins
p.df['Signal'] = pd.cut(p.df['PIP_Return_forward_looking_for_32mins_max'], bins, labels=[1, 2, 3, 4, 5, 6])
p.df[['Signal']]

#%%
print(p.df.groupby(p.df['Signal']).size())
p.df.groupby(p.df['Signal']).size().plot()


#%%
df = p.df[['Timestamp', 'Close', 'Signal']].copy()
dfDict = df.to_dict('split')
for i in range(1, 7):
    print('============='+str(i)+'==============')
    tradeDf = p.backtest(i, dfDict)
    tradeDf.describe()
    print('Number of trades: ' + str(len(tradeDf.index)))
    print('Final Balance: ' + str(tradeDf[-1:].Balance))
    tradeDf[['Balance']].plot()


#%% talib
import talib
df = p.df.loc[:, ['Open', 'High', 'Low','Close']].copy().dropna()
# tt = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
df['CDLTRISTAR'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
df[['CDLTRISTAR']].describe()
