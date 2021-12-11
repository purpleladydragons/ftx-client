import sys

import datetime as dt
import time
import pandas as pd
from pandas.testing import assert_frame_equal
import dateutil
import os

dirname = os.path.dirname(__file__)

sys.path.append('../ftx_client')
import ftx_client.api

def test_ticks_threaded():
    client = ftx_client.api.HelperClient(key=os.environ['API_KEY'], secret=os.environ['API_SECRET'], platform='com')

    start = dateutil.parser.parse('2021-12-01 00:00:00+00:00')
    end = start + dt.timedelta(seconds=60 * 60)

    df = client.get_historical_ticks_threaded('BTC-PERP', since=start, til=end)

    with open(dirname + '/fixtures/ticks.csv', 'r') as f:
        # We have to rename the column since the read_csv isn't smart enough to index the column automatically
        expected_df = pd.read_csv(f, index_col=0, parse_dates=True).rename(columns={'time.1': 'time'})
        assert_frame_equal(df, expected_df)


def test_prices_threaded():
    client = ftx_client.api.HelperClient(key=os.environ['API_KEY'], secret=os.environ['API_SECRET'], platform='com')

    start = dt.datetime.now() - dt.timedelta(days=5)
    then1 = time.time()
    df = client.get_prices('SOL-PERP', since_date=start, window_size_secs=60)
    print(time.time() - then1)
    print(len(df))


test_ticks_threaded()
