
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
p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017)_with_signals.csv'))
p.df.describe()



#%%
p.df['Max_PIP_Return_1'].loc[(p.df['Max_PIP_Return_1']>=16190)]
p.select_time_range('2011-10-25T00:00:00', '2011-11-02T00:00:00')['Close'].plot()

#%%
import numpy as np
mask = p.df['Max_PIP_Return_30'].loc[(p.df['Max_PIP_Return_30']>=1000) & (p.df['Max_PIP_Return_30']<=1001)]
len(mask.index)
# np.count_nonzero(mask)

#%%
import numpy as np
tmp = p.df.copy()
def convert_to_class(x):
    if x>=500:
        return
tmp['Max_PIP_Return_1_class'] = tmp['Max_PIP_Return_1'].apply(lambda x: )

#%% Maximum actual returns within following mins in the future
import numpy as np
arrKeys = []
for interval in intervals:
    arrKeys.append('Max_PIP_Return_'+str(interval))

#%% Get the percentile boundary
import numpy as np
percentiles = ['3', '16', '50', '84', '97']
results = {}
colToBeUsed = p.df['PIP_Return_forward_looking_for_32mins_max'].dropna()
for stat in percentiles:
    results[stat] = np.percentile(colToBeUsed, int(stat))
results


#%% Convert percentile into corresponding signal class: [
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

#%% Check the distribution of [1, 6] signal classes
signalDf = p.df.groupby(p.df['Signal']).size().to_frame()
signalDf.columns = ['Count']
signalDf.plot()
signalDf


#%% Get backtest results of all signals
allTrades = []
df = p.df[['Timestamp', 'Close', 'Signal']].copy()
dfDict = df.to_dict('split')
for i in range(1, 7):
    print('============= Signal_'+str(i)+' Performance ==============')
    tradeDf = p.backtest(i, dfDict)
    print('Final Balance: ' + str(float(tradeDf[-1:].Balance)))
    tradeDf[['Balance']].plot()
    allTrades.append(tradeDf)

#%% Findout which signal to be used, i.e. maximize profit while minimize drawdown
results = []
indexes = []
for i in range(0, 6):
    trade = allTrades[i]
    results.append({
        'Final Balance': float(trade[-1:].Balance),
        'Max Drawdown': float(trade.describe()[['Drawdown']].loc['min'])
    })
    indexes.append('Signal_' + str(i+1))
tmpDf = pd.DataFrame(results, index=indexes)
tmpDf[['Final Balance', 'Max Drawdown']].plot()
tmpDf

#%% 
# mergedDf.plot()
allTrades[5].to_csv(path.join(dataDir, 'signal6.csv'), sep=',', encoding='utf-8', index=False)

#%% talib
import talib
df = p.df.loc[:, ['Open', 'High', 'Low','Close', 'Signal']].copy().dropna()
df['CDLTRISTAR'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
df[['CDLTRISTAR']].describe()
df

#%% Get all pattern detection algorithm from talib
from inspect import getmembers, isfunction
functs = [o for o in getmembers(talib) if isfunction(o[1])]
featuresToBeUsed = []
for func in functs:
    funcName = func[0]
    if funcName.startswith('CDL'):
        print('Computing Pattern Features Using talib: ' + funcName)
        featuresToBeUsed.append(funcName)
        df[funcName] = getattr(talib, funcName)(df['Open'], df['High'], df['Low'], df['Close']) / 100
tmp = df.describe().T
tmp

#%% 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

tmp = df.copy()

tmp['Normalized_Close'] = ((tmp['Close'] - tmp['Close'].rolling(43200).min()) / (tmp['Close'].rolling(43200).max() - tmp['Close'].rolling(43200).min())).fillna(method='ffill')
featuresToBeUsed.append('Normalized_Close')

tmp.dropna(inplace=True)
train, test = train_test_split(tmp, test_size=0.2, shuffle=True)
print(len(tmp.index))
print(len(train.index))
print(len(test.index))

X_train = train[featuresToBeUsed].as_matrix()
Y_train = train['Signal'].as_matrix()

X_test = test[featuresToBeUsed].as_matrix()
Y_test = test['Signal'].as_matrix()


#%% import sklearn libs
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

#%% Stochastic gradient descent (SGD) learning algorithms (29.44%)
# sgd = linear_model.SGDClassifier(max_iter=20, tol=None)
# sgd.fit(X_train, Y_train)
# Y_pred = sgd.predict(X_test)
# sgd.score(X_train, Y_train)
# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# print(round(acc_sgd,2,), "%")

#%% Random Forest (87.65%)
# random_forest = RandomForestClassifier(n_estimators=100)
# print('start')
# random_forest.fit(X_train, Y_train)
# print('end')
# Y_prediction = random_forest.predict(X_test)
# print('score')
# random_forest.score(X_train, Y_train)
# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# print(round(acc_random_forest,2,), "%")
# Save Trained Random Forest Model
# joblib.dump(random_forest, path.join(dataDir, 'trained_random_forest.pkl'))

clf3 = joblib.load(path.join(dataDir, 'trained_random_forest.pkl'))
Y_prediction = clf3.predict(X_test)
Y_prediction

#%%
# out = Y_prediction - Y_test
# np.count_nonzero(out==0)
# np.count_nonzero(Y_test==6)


importances = pd.DataFrame({'feature':featuresToBeUsed,'importance':np.round(clf3.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()


#%% Logistic Regression (34.78%)
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print(round(acc_log,2,), "%")

#%% KNN (???)
# knn = KNeighborsClassifier(n_neighbors = 6)
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# print(round(acc_knn,2,), "%")

#%% Gaussian Naive Bayes (3.69%)
# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# print(round(acc_gaussian,2,), "%")

#%% Perceptron (33.59%)
# perceptron = Perceptron(max_iter=20)
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print(round(acc_perceptron,2,), "%")

#%% Linear SVC (34.79%)
# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print(round(acc_linear_svc,2,), "%")

#%% Decision Tree (87.68%)
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print(round(acc_decision_tree,2,), "%")