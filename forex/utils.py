import dateutil.parser
import pandas as pd

from datetime import datetime, timezone, tzinfo, timedelta

class simple_utc(tzinfo):
    def tzname(self,**kwargs):
        return "UTC"
    def utcoffset(self, dt):
        return timedelta(0)

def parseDateTime(str):
    return datetime.strptime(str, '%Y%m%d %H:%M:%S:%f').replace(tzinfo=simple_utc())

def parseISODateTime(str):
    return dateutil.parser.parse(str)

def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)

def substract_datetime(t, seconds):
    return t - datetime.timedelta(seconds=seconds)

# get_df_by_datetime_range(df, parseISODateTime('2008-01-02T00:00:00'), parseISODateTime('2017-01-03T00:00:00'))
def get_df_by_datetime_range(df, start_t, end_t):
    mask = (df['Timestamp'] >= start_t) & (df['Timestamp'] <= end_t)
    return df.loc[mask]

def set_df_Timestamp_as_datetime(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
    return df