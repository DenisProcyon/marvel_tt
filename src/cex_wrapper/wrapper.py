import time
import ccxt
import pandas as pd
from pathlib import Path

from logger.logger import Logger

class CexWrapper:
    """
    Wrapper for ccxt
    """
    
    MAX_LIMIT = 1000    

    def __init__(self, exchange_name: str = "binance", **exchange_kwargs) -> None:
        """
        :param exchange_name: exhange name
        :param exchange_kwargs: args for ccxt.Exchange constructor
        """
        self.exchange = self._get_exchange(exchange_name, **exchange_kwargs)

        self.logger = Logger(logger_name=f'cex_{exchange_name}', console_stream=True)

        self.logger.log(
            message=f'Wrapper for {self.exchange.name} initialized',
            level="info"
        )

        # Set up cache directory
        self.cache_dir = Path(__file__).parent.parent.parent / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_exchange(name: str, **kwargs) -> ccxt.Exchange:
        """
        Dynamic exchange mapper

        :raises ValueError: if no such exchane in ccxt
        """
        name = name.lower()
        if name not in ccxt.exchanges:
            raise ValueError(f'{name} not supported')
        
        return getattr(ccxt, name)(kwargs)

    @staticmethod
    def _to_ms(dt_str: str) -> int:
        return int(pd.Timestamp(dt_str, tz="UTC").timestamp() * 1000)

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        fn = f"{symbol.replace('/', '')}_{timeframe}.parquet"

        return self.cache_dir / fn

    def fetch_ohlcv(self, base: str, quote: str, timeframe: str, since_dt: str, until_dt: str | None = None) -> pd.DataFrame:
        """
        Get OHLCV data from exchange

        :param base: e.g.: 'BTC'
        :param quote: e.g.: 'USDT'
        :param timeframe: e.g.: '1d', '1h', '5m'
        :param since_dt: start of period
        :param until_dt: end of period. None by default, means until now
        """
        symbol = f"{base.upper()}/{quote.upper()}"
        path = self._cache_path(symbol, timeframe)

        # Check if cache exists
        if path.exists():
            data = pd.read_parquet(path)
            self.logger.log(message=f"Loaded {len(data)} rows from cache {path}", level="info")
        else:
            self.logger.log(message=f"Cache not found {path}. Fetching from exchange", level="info")
            if not self.exchange.markets:
                self.exchange.load_markets()

            if symbol not in self.exchange.symbols:
                raise ValueError(f"Symbol {symbol} not listed on {self.exchange.id}")

            since_ms = self._to_ms(since_dt)
            until_ms = self._to_ms(until_dt) if until_dt else self.exchange.milliseconds()
            tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
            rows = []

            while since_ms < until_ms:
                try:
                    chunk = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since_ms,
                        limit=self.MAX_LIMIT,
                    )
                except ccxt.BaseError as e:
                    self.logger.log(message=f"Error fetching data: {e}", level="error")
                    break
                if not chunk:
                    break
                rows.extend(chunk)
                since_ms = chunk[-1][0] + tf_ms
                self.logger.log(message=f"Fetched {len(chunk)} more. Total: {len(rows)}", level="info")
                time.sleep(self.exchange.rateLimit / 1000)
                if len(chunk) < self.MAX_LIMIT:
                    break

            data = (
                pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
            )

            data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
            data = data.set_index("datetime").drop(columns="timestamp").astype(float)
            data.to_parquet(path)
            
            self.logger.log(message=f"Saved {len(data)} rows to cache {path}", level="info")

        data = data.loc[since_dt:until_dt] if until_dt else data.loc[since_dt:]

        self.logger.log(
            level="info",
            message=f"Fetched {len(data)} bars for {symbol} {timeframe} (from {data.index.min().date()} to {data.index.max().date()})",
        )
        return data