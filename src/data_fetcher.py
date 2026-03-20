"""MEXC data fetcher for candle/kline data."""
import asyncio
import logging
import time
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from .config import MEXCConfig

logger = logging.getLogger(__name__)


class MEXCFetcher:
    """Fetches OHLCV candle data from MEXC API."""

    def __init__(self, config: MEXCConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.request_timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def _rate_limit(self):
        """Simple rate limiter."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            await asyncio.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    async def fetch_klines(
        self,
        interval: str = "5m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch kline/candlestick data from MEXC.

        Args:
            interval: Kline interval (5m, 15m, 60m, 4h, 1d)
            limit: Number of candles (max 500)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, close_time, quote_volume
        """
        await self._rate_limit()
        client = await self._get_client()

        # Map friendly interval names to MEXC format
        mexc_interval = self.config.intervals.get(interval, interval)

        params = {
            "symbol": self.config.symbol,
            "interval": mexc_interval,
            "limit": min(limit, self.config.max_klines),
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        try:
            response = await client.get(self.config.klines_endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning(f"No kline data returned for {interval}")
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close",
                "volume", "close_time", "quote_volume"
            ])

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = df[col].astype(np.float64)

            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.debug(f"Fetched {len(df)} klines for {interval}")
            return df

        except httpx.HTTPStatusError as e:
            logger.error(f"MEXC API error ({e.response.status_code}): {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch klines: {e}")
            raise

    async def fetch_multi_timeframe(
        self,
        intervals: Optional[list] = None,
        limit: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Fetch klines for multiple timeframes (single batch, no pagination).

        Used for real-time prediction where we only need the most recent candles.

        Args:
            intervals: List of intervals to fetch. Defaults to ["5m", "15m", "1h"]
            limit: Number of candles per timeframe

        Returns:
            Dict mapping interval -> DataFrame
        """
        if intervals is None:
            intervals = ["5m", "15m", "1h"]

        results = {}
        for interval in intervals:
            try:
                df = await self.fetch_klines(interval=interval, limit=limit)
                results[interval] = df
            except Exception as e:
                logger.error(f"Failed to fetch {interval} klines: {e}")
                results[interval] = pd.DataFrame()

        return results

    # Mapping from interval string to milliseconds per candle
    INTERVAL_MS = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "60m": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    # Mapping from interval string to minutes per candle
    INTERVAL_MINUTES = {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "60m": 60,
        "4h": 240,
        "1d": 1440,
    }

    async def fetch_historical_klines(
        self,
        interval: str = "5m",
        total_candles: int = 5000,
    ) -> pd.DataFrame:
        """Fetch large amounts of historical data by paginating with startTime+endTime.

        MEXC API requires BOTH startTime and endTime to fetch historical windows.
        We paginate forward in batch-sized windows until we have enough candles.

        Args:
            interval: Kline interval
            total_candles: Total number of candles to fetch

        Returns:
            DataFrame with all historical candles
        """
        interval_ms = self.INTERVAL_MS.get(interval, 5 * 60 * 1000)
        now_ms = int(time.time() * 1000)
        batch_size = self.config.max_klines  # 500

        # Start from (total_candles + buffer) candles ago
        start_time = now_ms - (total_candles + 50) * interval_ms

        all_dfs = []
        fetched = 0

        while fetched < total_candles and start_time < now_ms:
            # Set endTime to cover one batch window
            end_time = start_time + batch_size * interval_ms

            df = await self.fetch_klines(
                interval=interval,
                limit=batch_size,
                start_time=start_time,
                end_time=end_time,
            )

            if df.empty:
                # No data in this window, skip forward
                start_time = end_time + 1
                continue

            all_dfs.append(df)
            fetched += len(df)

            if len(df) < batch_size:
                # Partial batch -- advance past what we got
                latest_ts = int(df["timestamp"].iloc[-1].timestamp() * 1000)
                start_time = latest_ts + interval_ms
            else:
                # Full batch -- advance window
                start_time = end_time + 1

            logger.info(f"Fetched {fetched}/{total_candles} {interval} candles")

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Total historical {interval} candles fetched: {len(result)}")
        return result

    async def fetch_historical_multi_timeframe(
        self,
        intervals: list[str],
        train_candles_5m: int,
    ) -> dict[str, pd.DataFrame]:
        """Fetch paginated historical data for multiple higher timeframes.

        Calculates the proportional number of candles needed for each
        timeframe to cover the same calendar window as the 5m training data.

        For example, if train_candles_5m=43200 (~150 days):
          - 15m: 43200 * (5/15) = 14,400 candles (~150 days)
          - 1h:  43200 * (5/60) = 3,600 candles  (~150 days)

        Args:
            intervals: List of higher-timeframe intervals (e.g. ["15m", "1h"])
            train_candles_5m: Number of 5m candles being used for training

        Returns:
            Dict mapping interval -> DataFrame with full paginated historical data
        """
        results = {}

        for interval in intervals:
            try:
                # Calculate proportional candle count for same time window
                interval_minutes = self.INTERVAL_MINUTES.get(interval, 60)
                proportional_candles = max(
                    500,  # minimum useful amount
                    int(train_candles_5m * 5 / interval_minutes) + 50,  # +50 buffer for NaN drop
                )

                logger.info(
                    f"Fetching historical {interval} data: "
                    f"{proportional_candles} candles "
                    f"(~{proportional_candles * interval_minutes // 1440} days)"
                )

                df = await self.fetch_historical_klines(
                    interval=interval,
                    total_candles=proportional_candles,
                )
                results[interval] = df

                if not df.empty:
                    logger.info(
                        f"Historical {interval}: {len(df)} candles, "
                        f"from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
                    )
                else:
                    logger.warning(f"Historical {interval}: no data returned")

            except Exception as e:
                logger.error(f"Failed to fetch historical {interval} klines: {e}")
                results[interval] = pd.DataFrame()

        return results

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
