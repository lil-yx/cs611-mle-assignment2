import pandas as pd


def month_starts(start_date, end_date):
    """First day of every month between start and end, inclusive, as YYYY-MM-DD strings."""
    return pd.date_range(start_date, end_date, freq="MS").strftime("%Y-%m-%d").tolist()
