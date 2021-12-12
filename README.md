# FTX Client

A python wrapper and client for the FTX API.

In addition to support basic routes, this client also provides helper methods to quickly download paginated results.

## Usage

```python3
import datetime as dt
from ftx_client.api import HelperClient

# point client at ftx.com (instead of ftx.us)
client = HelperClient(key='abc', secret='xyz', platform='com')

now = dt.datetime.now()
since = now - dt.timedelta(days=7)

# get 1minute OHLC candles for BTC/USDT from 7 days ago to now
client.get_historical_prices('BTC/USDT', since_date=since, end_date=now, window_size_secs=60)  
```
