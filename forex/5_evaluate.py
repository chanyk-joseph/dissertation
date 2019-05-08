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


currencyPairs = ['AUDUSD', 'EURGBP', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'XAUUSD']
if len(sys.argv) > 1:
    currencyPairs = [sys.argv[1]]
print(currencyPairs)

currencyPair = 'USDJPY'
# for currencyPair in currencyPairs:
from sklearn.externals import joblib

directory = path.join(dataDir, currencyPair)

tmp = pd.read_csv(path.join(directory, 'features_evaluation_stage_2_test_set.csv'))

random_forest = joblib.load(path.join(directory, 'features_evaluation_stage_2_trained_random_forest.pkl'))

featuresToBeUsed = tmp.columns.values[[colName.startswith('patterns_') for colName in tmp.columns.values]]
X_test = tmp[featuresToBeUsed].values
Y_test = tmp['BuyHoldSell'].values

Y_prediction = random_forest.predict(X_test)
Y_prediction

#%%
totalNumRecords = len(tmp.index)
buyRecords = len(Y_test[Y_test == 1])
sellRecords = len(Y_test[Y_test == -1])

print('Number of Buy(1) actually: ' + str(buyRecords))
print('Number of predicted Buy(1): ' + str(len(Y_prediction[Y_prediction == 1])))

print('Number of Buy(1) actually: ' + str(sellRecords))
print('Number of predicted Buy(1): ' + str(len(Y_prediction[Y_prediction == -1])))

#%%
tmp2 = Y_prediction[Y_test == 1]
print('Total Long: ' + str(len(tmp2)))
print('Incorrect: ' + str(len(tmp2[tmp2 != 1])))
print('%: ' + str(len(tmp2[tmp2 != 1]) / len(tmp2) * 100))



#%%
tmp2 = Y_prediction[Y_test == -1]
print('Total Short: ' + str(len(tmp2)))
print('Incorrect: ' + str(len(tmp2[tmp2 != -1])))
print('%: ' + str(len(tmp2[tmp2 != -1]) / len(tmp2) * 100))

#%%
tmp2 = Y_prediction[Y_test == 0]
print('Total Hold: ' + str(len(tmp2)))
print('Incorrect: ' + str(len(tmp2[tmp2 != 0])))
print('%: ' + str(len(tmp2[tmp2 != 0]) / len(tmp2) * 100))

