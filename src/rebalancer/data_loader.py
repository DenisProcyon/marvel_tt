from typing import Dict, List, Optional

import pandas as pd

from cex_wrapper.wrapper import CexWrapper

class CexDataLoader:
    """
    Data loader for CEX exchanges, implementation of CexWrapper class
    """
    def __init__(self, exchange_name: str, timeframe: str = "1d") -> None:
        self.wrapper = CexWrapper(exchange_name=exchange_name)
        self.timeframe = timeframe

    def get_top_volume_symbols(self, n: int = 50) -> List[str]:
        """
        Getting symbols by top volume
        """
        tickers = self.wrapper.exchange.fetch_tickers()
        df = (
            pd.DataFrame.from_dict(tickers, orient="index")
            .loc[lambda d: d.index.str.endswith("USDT") & ~d.index.str.startswith("USD")]
            .assign(volume_usd=lambda d: d["quoteVolume"])
            .sort_values("volume_usd", ascending=False)
            .head(n)
        )
        return df["symbol"].tolist()

    def fetch_history(
        self, symbols: List[str], since: str, until: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            base, quote = sym.split("/")
            ohlcv = self.wrapper.fetch_ohlcv(
                base, quote, self.timeframe, since_dt=since, until_dt=until
            )
            data[sym] = ohlcv
        return data