import hmac
import json
import time
import datetime
import pandas as pd
import queue
from concurrent.futures import ThreadPoolExecutor
import concurrent
import logging
from requests import Request, Session
from typing import Optional


class RestClient:
    us_api_url = 'https://ftx.us/api'
    com_api_url = 'https://ftx.com/api'

    def __init__(self, key, secret, platform='us'):
        if platform == 'us':
            self.api_url = RestClient.us_api_url
        else:
            self.api_url = RestClient.com_api_url
        self._session = Session()
        self._api_key = key
        self._api_secret = secret

    def _make_request(self, verb, endpoint, params={}, json_body={}):
        """
        Make request to FTX US Rest API

        :param verb: HTTP Verb (GET, POST etc)
        :param endpoint: API endpoint
        :param params: Query params
        :param json_body: JSON payload params
        :return:
        """
        ts = int(time.time() * 1000)
        request = Request(verb, f'{self.api_url}/{endpoint}', params=params, json=json_body)
        prepared = request.prepare()
        signature_payload = f'{ts}{prepared.method}{prepared.path_url}'.encode()
        if prepared.body:
            signature_payload += prepared.body

        signature = hmac.new(self._api_secret.encode(), signature_payload, 'sha256').hexdigest()

        prepared.headers['FTXUS-KEY'] = self._api_key
        prepared.headers['FTXUS-SIGN'] = signature
        prepared.headers['FTXUS-TS'] = str(ts)

        return prepared

    def _send_req(self, request):
        resp = self._session.send(request)
        data = json.loads(resp.text)
        return data

    def get_markets(self):
        req = self._make_request('GET', 'markets')
        return self._send_req(req)

    def get_futures_stats(self, market):
        """

        :param market: relevant market / future, e.g BTC-PERP
        :return:
        """
        req = self._make_request('GET', f'futures/{market}/stats')
        resp = self._send_req(req)
        return resp

    def get_funding_rates(self, market, start, end):
        """

        :param market: market/future you're interested in, e.g BTC-PERP
        :param start: start of the historical window you're interested in, in seconds-based timestamp
        :param end: end of the window
        :return: 
        """
        endpoint = f'funding_rates'
        params = {
            'future': market,
            'start_time': start,
            'end_time': end,
        }
        req = self._make_request('GET', endpoint, params)
        resp = self._send_req(req)
        return resp

    def get_historical_prices(self, market, start, end, resolution):
        """
        GET /markets/{market_name}/candles?resolution={resolution}&limit={limit}&start_time={start_time}&end_time={end_time}

        :param market: market that you care about, e.g "BTC/USD"
        :param start: start of window in seconds-based timestamp
        :param end: end of window in seconds-based timestamp
        :param resolution: how to sample the given window, in seconds, e.g 60 to get 1 minute candles
        :return: DataFrame-compatible table of OHLCV data
        """
        params = {
            'start_time': start,
            'end_time': end,
            'resolution': resolution
        }
        endpoint = f'markets/{market}/candles'
        req = self._make_request('GET', endpoint, params)
        resp = self._send_req(req)
        return resp

    def get_ticks(self, market, start, end):
        """
        GET /markets/{market_name}/trades?limit={limit}&start_time={start_time}&end_time={end_time}

        :param market:
        :param start:
        :param end:
        :return:
        """
        params = {
            'start_time': start,
            'end_time': end,
            'limit': 100
        }
        endpoint = f'markets/{market}/trades'
        req = self._make_request('GET', endpoint, params)
        resp = self._send_req(req)
        return resp

    def get_orderbook(self, market):
        req = self._make_request('GET', f'markets/{market}/orderbook')
        resp = self._send_req(req)
        return resp

    def get_order_status(self, order_id):
        req = self._make_request('GET', f'orders/{order_id}')
        resp = self._send_req(req)
        return resp

    def get_balances(self):
        req = self._make_request('GET', 'wallet/balances')
        resp = self._send_req(req)
        return resp

    def place_order(self, **kwargs):
        try:
            market = kwargs['market']
            side = kwargs['side']
            price = kwargs['price']
            typ = kwargs['type']
            size = kwargs['size']
            ioc = kwargs.get('ioc', False)
        except KeyError as e:
            logging.error('Missing required param to send order', e)
            return None
        json_body = {
            'market': market,
            'side': side,
            'price': price,
            'size': size,
            'type': typ,
            'ioc': ioc,
            'clientId': None
        }
        req = self._make_request(
            verb='POST',
            endpoint='orders',
            params={},
            json_body=json_body
        )
        resp = self._send_req(req)
        return resp


class HelperClient(RestClient):
    def __init__(self, key, secret, platform='us'):
        super().__init__(key, secret, platform)

    def _get_prices_helper_threaded(self, coin, since_date, window_size_secs, verbose=False):
        tildate = datetime.datetime.now()
        prices_til = int(time.mktime(tildate.timetuple()))
        since = int(time.mktime(since_date.timetuple()))
        prices_cum = []

        # (end - start) / size = # of points needed
        # #pts / 10k = # requests

        max_points_per_request = 1500

        points_needed = (prices_til - since) / window_size_secs
        logging.info(f'Making {max(1, points_needed / max_points_per_request)} requests')
        now_secs = tildate.timestamp()

        prices_cum = {}

        def download_range(range):
            start, end = range
            prices = self.get_historical_prices(f'{coin}', start, end, window_size_secs)
            prices_cum[start] = prices

        ranges = []
        for start in range(since, prices_til, window_size_secs * max_points_per_request):
            end = min(now_secs, start + window_size_secs * max_points_per_request)
            ranges.append((start, end))

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            executor.map(download_range, ranges)

        sorted_prices = [y[1] for y in sorted(list(prices_cum.items()), key=lambda x: x[0])]
        return sorted_prices

    def _get_prices_helper(self, coin, since_date, window_size_secs, verbose=False):
        tildate = datetime.datetime.now()
        prices_til = int(time.mktime(tildate.timetuple()))
        since = int(time.mktime(since_date.timetuple()))
        prices_cum = []

        # (end - start) / size = # of points needed
        # #pts / 10k = # requests

        max_points_per_request = 1500

        points_needed = (prices_til - since) / window_size_secs
        logging.info(f'Making {max(1, points_needed / max_points_per_request)} requests')
        now_secs = tildate.timestamp()
        for start in range(since, prices_til, window_size_secs * max_points_per_request):
            end = min(now_secs, start + window_size_secs * max_points_per_request)
            if verbose:
                start_hum = datetime.datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                end_hum = datetime.datetime.utcfromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
                logging.info('looping', start_hum, 'to', end_hum)
            time.sleep(1)
            prices = self.get_historical_prices(f'{coin}', start, end, window_size_secs)
            if verbose:
                if 'result' in prices:
                    logging.info('fetched', len(prices['result']), 'results')
                else:
                    logging.info('error?', prices)
            prices_cum.append(prices)

        return prices_cum

    def _combine_prices_into_df(self, prices_cum):
        pdfs = []
        for prices in prices_cum:
            pdf = pd.DataFrame(prices['result'])
            pdfs.append(pdf)
        pdf = pd.concat(pdfs)
        pdf.index = pd.to_datetime(pdf['startTime'].sort_values())
        return pdf

    def get_prices(self, coin, since_date, window_size_secs, verbose=False):
        """

        :param coin:
        :param since_date: DateTime
        :param window_size_secs:
        :param verbose:
        :return:
        """
        prices = self._get_prices_helper_threaded(coin, since_date, window_size_secs, verbose)
        return self._combine_prices_into_df(prices)

    def get_historical_ticks_threaded(self, market, since, til):
        """
        Request tick data in parallel. Since we can't know ahead of time how many ticks occur in a given window,
        we use a thread pool and a queue to add and remove tasks.

        The smallest resolution that FTX supports is 1 second. If there are >100 ticks in a second,
        then we can only retrieve the first 100.

        :param market:
        :param since:
        :param til:
        :return:
        """
        cum_ticks = {}
        max_threads = 80

        def _thread_action(window_start, window_end):
            # TODO some error handling around missing 'result' key
            ticks = self.get_ticks(market, start=window_start, end=window_end)['result']
            # TODO would it be more efficient to just save the first 100 in and then put the last tick's time in as a new task? rather than repeat work for both?
            # if too many results, split the window and put them back on the queue
            if len(ticks) >= 100 and window_end - window_start > 1:
                new_end = (window_start + window_end) // 2
                q.put((window_start, new_end))
                q.put((new_end, window_end))
            else:
                cum_ticks[window_start] = ticks

        since_ts = int(time.mktime(since.timetuple()))
        til_ts = int(time.mktime(til.timetuple()))

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            q = queue.Queue(1000)

            q.put((since_ts, til_ts))
            last_update = time.time()

            # TODO this exit logic should be improved
            while not (q.empty() and time.time() - last_update > 15):
                try:
                    item = q.get(timeout=5)
                    executor.submit(_thread_action, item[0], item[1])
                except queue.Empty:
                    break

        # TODO ugly
        data = [y[1] for y in sorted([x for x in cum_ticks.items()], key=lambda kv: kv[0])]
        dfs = []
        for tick in data:
            df = pd.DataFrame(tick)
            # each request orders by time desc, so you get like: 5-4-3-2-1-9-8-7-6
            # so reverse each df first before appending
            dfs.append(df.iloc[::-1])
        if len(dfs) > 0:
            df = pd.concat(dfs)
            df.index = pd.to_datetime(df['time'])
            return df

    def _get_paginated_results(self, market: str, endpoint_func, start: datetime, end: datetime, verbose) -> Optional[
        pd.DataFrame]:
        """
        :param market: market symbol, e.g BTC/USD
        :param endpoint_func: function that returns results from a single page for given symbol and window
        :param start: the earliest point to capture data from
        :param end: the latest point to capture data from
        :return: a dataframe containing the results
        """

        # the FTX API expects seconds-based timestamps, so we conver the datetimes
        since_ts = int(time.mktime(start.timetuple()))
        til_ts = int(time.mktime(end.timetuple()))

        # we'll accumulate the paginated results in this array
        cum_ticks = []

        max_data_size = 100  # FTX supports up to 100 datapoints per page
        max_window_size = 60 * 60 * 24
        window_start = since_ts
        window_size = max_window_size
        window_end = window_start + window_size

        # TODO probably better to include the first 100 points and use that window's end as new start time regardless of result

        # We slide the window forward through time. If we grab too many datapoints, then we halve the window size
        # repeatedly until it's no longer too large. Each time we successfully download a page, we double the window size
        while window_start < til_ts:
            if verbose:
                start_hum = datetime.datetime.utcfromtimestamp(window_start).strftime('%Y-%m-%d %H:%M:%S')
                end_hum = datetime.datetime.utcfromtimestamp(window_end).strftime('%Y-%m-%d %H:%M:%S')
                logging.info('Collecting funding rates from', start_hum, 'to', end_hum)
            ticks = endpoint_func(market, start=window_start, end=window_end)
            if len(ticks['result']) >= max_data_size and window_size > 1:
                if verbose:
                    logging.info('too many results', len(ticks['result']))
                window_size = max(1, window_size / 2)
                window_end = window_start + window_size
                continue
            else:
                cum_ticks.append(ticks['result'])
                window_size = min(max_window_size, window_size * 2)
                window_start = window_end
                window_end = window_start + window_size

        dfs = []
        for tick in cum_ticks:
            df = pd.DataFrame(tick)
            # the page results are in ascending order
            # but the points in each page are ordered by descending time
            # so we reverse each df first before appending
            dfs.append(df.iloc[::-1])
        if len(dfs) > 0:
            df = pd.concat(dfs)
            df.index = pd.to_datetime(df['time'])
            return df
        return None

    def get_historical_funding_rates(self, market: str, since: datetime, til: datetime, verbose=False) -> Optional[
        pd.DataFrame]:
        """
        Get all the hourly funding rates for a given market in a given window of time.

        :param market: market symbol, e.g BTC/USD
        :param since:
        :param til:
        :param verbose:
        :return:
        """
        return self._get_paginated_results(market, self.get_funding_rates, since, til, verbose)

    def get_historical_ticks(self, market: str, since: datetime, til: datetime, verbose=False) -> Optional[
        pd.DataFrame]:
        """
        Get all the ticks for a given market in a given window of time.

        :param market: market symbol, e.g BTC/USD
        :param since:
        :param til:
        :param verbose:
        :return:
        """
        return self._get_paginated_results(market, self.get_ticks, since, til, verbose)
