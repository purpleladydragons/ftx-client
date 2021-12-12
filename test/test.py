import sys

import datetime as dt
import pandas as pd
from pandas.testing import assert_frame_equal
import dateutil
import os

dirname = os.path.dirname(__file__)

sys.path.append("ftx_client")
import api


def test_ticks_threaded():
    client = api.HelperClient(key="", secret="", platform="com")

    start = dateutil.parser.parse("2021-12-01 00:00:00+00:00")
    end = start + dt.timedelta(seconds=60 * 60)

    df, errors = client.get_historical_ticks_threaded(
        "BTC-PERP", since_date=start, end_date=end
    )

    with open(dirname + "/fixtures/ticks.csv", "r") as f:
        # We have to rename the column since the read_csv isn't smart enough to index the column automatically
        expected_df = pd.read_csv(f, index_col=0, parse_dates=True).rename(
            columns={"time.1": "time"}
        )
        assert_frame_equal(df, expected_df)


def test_prices_threaded():
    client = api.HelperClient(key="", secret="", platform="com")

    start = dateutil.parser.parse("2021-12-01 00:00:00+00:00")
    end = start + dt.timedelta(seconds=60 * 60 * 15)

    df, errors = client.get_historical_prices(
        "BTC-PERP", since_date=start, end_date=end, window_size_secs=15
    )

    with open(dirname + "/fixtures/candles.csv", "r") as f:
        # We have to rename the column since the read_csv isn't smart enough to index the column automatically
        expected_df = pd.read_csv(f, index_col=0, parse_dates=True).rename(
            columns={"startTime.1": "startTime"}
        )
        assert_frame_equal(df, expected_df)


test_ticks_threaded()
test_prices_threaded()
