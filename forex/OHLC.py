import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

class OHLC:
    def __init__(self, csv_path):
        self.csv_path = csv_path

        self.df = self._init_df(csv_path)

    def _init_df(self, csv_path):
        df = pd.read_csv(csv_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
        df.index = pd.DatetimeIndex(df['Timestamp'])
        return df

    def set_df(self, df):
        self.df = None
        self.df = df
        return self

    def merge_df(self, df):
        self.df = pd.concat([self.df, df], axis=1) # https://stackoverflow.com/questions/40468069/merge-two-dataframes-by-index
        return self

    def print(self):
        print(self.df)
        return self

    def save(self, filepath):
        self.df.to_csv(filepath, sep=',', encoding='utf-8', index=False)
        return self

    # p.plot_fields('Close', [], parseISODateTime('2010-01-01T00:00:00'), parseISODateTime('2010-12-31T00:00:00'))
    def plot_fields(self, leftCol, rightCols=[], start_t=None, end_t=None):
        df = None
        if start_t != None and end_t != None:
            mask = (self.df['Timestamp'] >= start_t) & (self.df['Timestamp'] <= end_t)
            df = self.df.loc[mask]
        else:
            df = self.df

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Datetime')
        ax1.set_ylabel(leftCol, color=color)
        ax1.plot(df[leftCol], label=leftCol, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        if len(rightCols) > 0:
            ax2 = ax1.twinx()
            for col in rightCols:
                ax2.plot(df[col], label=col)
            ax2.legend()
        fig.tight_layout()
        plt.show()

    def get_df_with_resolution(self, resolutionStr): # 1S, 1min, 1H, 1D
        validDateTimes = self.df.index.floor('H')

        df = self.df.resample(resolutionStr).pad() # pad: fill NaN with previous value 
        df['Timestamp'] = df.index

        allDateTimesAfterResample = df.index.floor('H')
        df = df[~allDateTimesAfterResample.isin(allDateTimesAfterResample.difference(validDateTimes))]

        return df

    def get_normalized_price(self, rollingPeriods, unit):
        # cl = self.df['Close'].values.copy().reshape(-1, 1)
        # scaler = MinMaxScaler()
        # for di in range(0, len(cl), smoothing_window_size):
        #     scaler.fit(cl[di:di+smoothing_window_size])
        #     cl[di:di+smoothing_window_size] = scaler.transform(cl[di:di+smoothing_window_size])

        def apply_ema_smoothing(arr):
            # Now perform exponential moving average smoothing
            # So the data will have a smoother curve than the original ragged data
            EMA = 0.0
            gamma = 0.1
            for ti in range(len(arr)):
                if np.isnan(arr[ti]):
                    continue
                    if ti >= 1 and (not np.isnan(arr[ti-1])):
                        arr[ti] = arr[ti-1]
                        continue
                    else:
                        continue
                EMA = gamma*arr[ti] + (1-gamma)*EMA
                arr[ti] = EMA
            return arr
        def normalize(df):
            for period in rollingPeriods:
                df['Close_Normalized_By_Past_' + str(period)+unit] = ((df['Close'] - df['Close'].rolling(period).min()) / (df['Close'].rolling(period).max() - df['Close'].rolling(period).min())).fillna(method='ffill')
                df['Close_Normalized_By_Future_' + str(period)+unit] = ((df['Close'] - df['Close'][::-1].rolling(period).min()[::-1]) / (df['Close'][::-1].rolling(period).max()[::-1] - df['Close'][::-1].rolling(period).min()[::-1])).fillna(method='bfill')
                df['Close_Normalized_By_Past_' + str(period)+unit] = apply_ema_smoothing(df['Close_Normalized_By_Past_' + str(period)+unit].copy().values.reshape(-1, 1))
                df['Close_Normalized_By_Future_' + str(period)+unit] = apply_ema_smoothing(df['Close_Normalized_By_Future_' + str(period)+unit].copy().values.reshape(-1, 1))
        
        df = self.df.loc[:, ['Close']].copy()
        normalize(df)
        df = df.drop('Close', axis=1)
        return df

    def get_mins_returns_cols(self, rollingPeriods, unit):
        def calculate_log_returns(df, rollingPeriods):
            df['LogReturn_1' + unit] = np.log(df.Close).diff()

            for period in rollingPeriods:
                if period == 1:
                    df['LogReturn_forward_'+str(period)+unit+'_sum'] =  df['LogReturn_1'+unit].shift(-1)
                    continue
                df['LogReturn_forward_'+str(period)+unit+'_sum'] = df['LogReturn_1'+unit][::-1].shift(1).rolling(period).sum()[::-1] * 10000.0

            for period in rollingPeriods:
                if period == 1:
                    df['LogReturn_forward_'+str(period)+unit+'_min'] =  df['LogReturn_forward_'+str(period)+unit+'_sum'].shift(-1)
                    df['LogReturn_forward_'+str(period)+unit+'_max'] =  df['LogReturn_forward_'+str(period)+unit+'_sum'].shift(-1)
                    continue
                df['LogReturn_forward_'+str(period)+unit+'_min'] = df['LogReturn_forward_'+str(period)+unit+'_sum'][::-1].shift(1).rolling(period).min()[::-1]
                df['LogReturn_forward_'+str(period)+unit+'_max'] = df['LogReturn_forward_'+str(period)+unit+'_sum'][::-1].shift(1).rolling(period).max()[::-1]
            return df

        def calculate_pips_return(df, rollingPeriods):
            df['PIP_Return_1'+unit] = df.Close.diff() * 10000
            for period in rollingPeriods:
                if period == 1:
                    df['PIP_Return_forward_looking_for_'+str(period)+unit+'_sum'] =  df['PIP_Return_1'+unit].shift(-1)
                    continue
                df['PIP_Return_forward_looking_for_'+str(period)+unit+'_sum'] = df['PIP_Return_1'+unit][::-1].shift(1).rolling(period).sum()[::-1]

            for i, period in enumerate(rollingPeriods):
                print("period: "+str(period))
                colToBeUsed = []
                for j in range(i+1):
                    colToBeUsed.append('PIP_Return_forward_looking_for_'+str(rollingPeriods[j])+unit+'_sum')
                print("colToBeUsed: " + ",".join(colToBeUsed))
                df['PIP_Return_forward_looking_for_'+str(period)+unit+'_min'] = df.loc[:, colToBeUsed].min(axis=1)
                df['PIP_Return_forward_looking_for_'+str(period)+unit+'_max'] = df.loc[:, colToBeUsed].max(axis=1)

                vals = df['PIP_Return_forward_looking_for_'+str(period)+unit+'_max'].values.copy().reshape(-1, 1)
                scaler = MinMaxScaler()
                smoothing_window_size = 10000
                for di in range(0, len(vals), smoothing_window_size):
                    scaler.fit(vals[di:di+smoothing_window_size])
                    vals[di:di+smoothing_window_size] = scaler.transform(vals[di:di+smoothing_window_size])
                df['Normalized_PIP_Return_forward_looking_for_'+str(period)+unit+'_max'] = vals
            return df

        df = calculate_log_returns(self.df.loc[:, ['Close']].copy(), rollingPeriods)
        df = calculate_pips_return(df, rollingPeriods)
        print(len(df.index))
        for i, period in enumerate(rollingPeriods):
            if period == 1:
                df['is_all_forward_looking_increasing_within_1'+unit] = np.where(df['PIP_Return_forward_looking_for_1'+unit+'_sum'] > 0, 1, 0)
                continue

            df['is_all_forward_looking_increasing_within_'+str(period)+unit] = 1
            cols = []
            for i, previousPeriod in enumerate(rollingPeriods[0:i+1]):
                cols.append('PIP_Return_forward_looking_for_'+str(previousPeriod)+unit+'_sum')
                if i >= 1:
                    df['is_all_forward_looking_increasing_within_'+str(period)+unit] = np.where(df['is_all_forward_looking_increasing_within_'+str(period)+unit] & (df[cols[i]] > df[cols[i-1]]), 1, 0)

        # buy_signal = df['is_all_forward_looking_increasing_within_1min'].where(lambda x : x == 1).dropna()
        # print(len(df.index))
        # print(len(buy_signal))
        df = df.drop('Close', axis=1)
        return df