class Strategy:
    def __init__(self, name="Strategy Base Class"):
        self.name = name
        self.params = {}

    # get_dataframe_by_time_arange(df, parseISODateTime('2008-01-02T00:00:00'), parseISODateTime('2017-01-03T00:00:00'))
    def get_dataframe_by_time_arange(self, df, startTime, endTime):
        mask = (df['Timestamp'] >= startTime) & (df['Timestamp'] <= endTime)
        return df.loc[mask]

    def set_param(self, params):
        self.params = params