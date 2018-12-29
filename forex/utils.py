from datetime import datetime, timezone

def parseDateTimeStr(str):
    # 20080101 00:00:00:000
    return datetime.strptime(str, '%Y%m%d %H:%M:%S:%f')

def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)