import datetime
import talib
import numpy as np

class TimeRangeStrategies:
    def __init__(self, duration_in_seconds):
        super().__init__()
        self.duration_in_seconds = duration_in_seconds

    # get_dataframe_by_time_arange(df, parseISODateTime('2008-01-02T00:00:00'), parseISODateTime('2017-01-03T00:00:00'))
    def get_dataframe_by_time_arange(self, df, startTime, endTime):
        mask = (df['Timestamp'] >= startTime) & (df['Timestamp'] <= endTime)
        return df.loc[mask]

    def get_dataframe_by_time_delta(self, df, timeNow, num_of_seconds_before):
        from_datetime = timeNow - datetime.timedelta(seconds=num_of_seconds_before)
        return self.get_dataframe_by_time_arange(df, from_datetime, timeNow)

    def SMA_Crossover(self, df, end_datetime):
        def calculate(short_in_seconds, long_in_seconds):
            short_df = self.get_dataframe_by_time_delta(df, end_datetime, short_in_seconds)
            long_df = self.get_dataframe_by_time_delta(df, end_datetime, long_in_seconds)
            if len(short_df.index) == 0 or len(long_df.index) == 0:
                return np.nan
            short_val = talib.MA(short_df["Close"], len(short_df.index), matype=0)
            long_val = talib.MA(long_df["Close"], len(long_df.index), matype=0)
            

        from_datetime = end_datetime - datetime.timedelta(seconds=self.duration_in_seconds)
        sub_df = self.get_dataframe_by_time_arange(df, from_datetime, end_datetime)
        if len(sub_df.index) == 0:
            return np.nan
        result = talib.MA(sub_df["Close"], len(sub_df.index), matype=0)
        if len(result) == 0 or result.iloc[-1] == np.nan:
            return np.nan
        return result.iloc[-1]