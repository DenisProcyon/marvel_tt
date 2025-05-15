from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cex_wrapper.wrapper import CexWrapper
from logger.logger import Logger

class BaseStrategy:
    """
    Base class for all strategies
    """
    def compute_target_usd(self, window: pd.DataFrame, px_today: pd.Series) -> pd.Series:
        raise NotImplementedError

class TopNEqualDollarStrategy(BaseStrategy):
    """
    Regulat buy and hol dstrategy stated in the proposed task.
    """
    def __init__(self, n_top: int = 5, usd_per_strat: float = 1_000) -> None:
        self.n_top = n_top
        self.usd_per_strat = usd_per_strat

    def compute_target_usd(self, window: pd.DataFrame, px_today: pd.Series) -> pd.Series:
        # Calculate the performance of each asset in the window
        perf = (window.iloc[-1] / window.iloc[0] - 1).sort_values(ascending=False)

        # Select the top N assets based on performance
        top = perf.head(self.n_top).index
        tgt = pd.Series(0.0, index=px_today.index)
        tgt.loc[top] = self.usd_per_strat
        
        return tgt
    
class DynamicEWMAStrategy(BaseStrategy):
        """
        Dynamic strategy based on the Exponential Weighted Moving Average
        """
        def __init__(self, n_top: int = 5, usd_per_strat: float = 1_000, span: int = 25) -> None:
            self.n_top = n_top
            self.usd_per_strat = usd_per_strat
            self.span = span

        def compute_target_usd(self, window: pd.DataFrame, px_today: pd.Series) -> pd.Series:
            """
            Calculate the target USD allocation based on the Exponential Weighted Moving Average
            """
            ewma = window.ewm(span=self.span, adjust=False).mean().iloc[-1]
            
            # Calculate the deviation of price to the EWMA
            deviation = ((px_today - ewma) / ewma).abs()
            deviation = deviation.replace([np.inf, -np.inf], np.nan).dropna()

            # Select the top N assets based on deviation
            top = deviation.sort_values(ascending=False).head(self.n_top).index

            tgt = pd.Series(0.0, index=px_today.index)
            tgt.loc[top] = self.usd_per_strat
            
            return tgt

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

@dataclass
class BacktestResult:
    """
    Backtest result data class
    """
    equity_curve: pd.Series
    invested_curve: pd.Series
    total_fee: float
    total_pnl: pd.Series
    traded_volume: float

class Rebalancer:
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        strategy: BaseStrategy,
        lookback_days: int = 30,
        rebalance_freq_days: int = 7,
        fee_rate: float = 0.001,
        logger: Optional[Logger] = None) -> None:
        """
        :param price_data: Dictionary of price dataframes for each symbol
        :param strategy: Strategy to use for rebalancing
        :param lookback_days: Number of days to look back for strategy
        :param rebalance_freq_days: Frequency of rebalancing in days
        :param fee_rate: Transaction fee rate
        :param logger: Logger instance optional
        """

        self.strategy = strategy
        self.lookback_days = lookback_days
        self.rebalance_freq_days = rebalance_freq_days
        self.fee_rate = fee_rate
        self.logger = logger or Logger("rebalancer", console_stream=False)
        self.prices = self._build_price_matrix(price_data)

    def _log(self, msg: str) -> None:
        self.logger.log(msg, level="info")

    @staticmethod
    def _build_price_matrix(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build a price matrix from the price data dictionary
        """
        closes = {s: d["close"] for s, d in price_data.items()}
        return pd.concat(closes, axis=1).sort_index().ffill()

    def run(self) -> BacktestResult:
        """
        Run the backtest
        """
        dates = self.prices.index.normalize().unique()
        rebals = dates[self.lookback_days :: self.rebalance_freq_days]

        qty = pd.Series(0.0, index=self.prices.columns)
        cash = invested = total_fee = traded_volume = 0.0
        cost_basis_usd = pd.Series(0.0, index=self.prices.columns)
        realized_pnl = pd.Series(0.0, index=self.prices.columns)
        equity_curve, invested_curve = [], []

        self._log(
            f"Start back test: {dates[0].date()} ➜ {dates[-1].date()}"
        )
        self._log(
            f"Universe size={len(self.prices.columns)}, "
            f"strategy={self.strategy.__class__.__name__}, "
            f"lookback={self.lookback_days} days, "
            f"rebalance every {self.rebalance_freq_days} days"
        )

        for d in dates:
            px_today = self.prices.loc[d]

            # If we have a position, update the cost basis
            if d in rebals:

                window = self.prices.loc[
                    d - timedelta(days=self.lookback_days) : d - timedelta(days=1)
                ]

                # Calculate current alloc
                pre_val = (qty * px_today).loc[qty > 0]
                self._log(
                    "Before rebalance | Positions: "
                    + (", ".join(f"{s}: {pre_val[s]:,.2f}$" for s in pre_val.index) or "— none —")
                )

                target_usd = (
                    self.strategy.compute_target_usd(window, px_today)
                    .reindex(px_today.index)
                    .fillna(0.0)
                )
                target_qty = (target_usd / px_today).fillna(0.0)

                # Calculate the difference between target and current position
                dq = target_qty - qty
                selling, buying = dq.clip(upper=0).abs(), dq.clip(lower=0)

                realized_pnl += ((px_today * selling) - (cost_basis_usd / qty * selling).fillna(0))
                cost_basis_usd -= (cost_basis_usd / qty * selling).fillna(0)
                cost_basis_usd += (px_today * buying)
                qty = target_qty

                buy_usd = (buying.abs() * px_today).sum()
                sell_usd = (selling * px_today).sum()
                traded_usd = buy_usd + sell_usd
                traded_volume += traded_usd
                fee_paid = traded_usd * self.fee_rate
                total_fee += fee_paid

                net_cash = cash + sell_usd
                need = max(0.0, buy_usd + fee_paid - net_cash)

                # If we need more cash, we add it
                if need:
                    cash += need
                    invested += need
                cash = cash - buy_usd + sell_usd - fee_paid

                pos_val = (qty * px_today).loc[qty > 0]
                pos_str = (", ".join(f"{s}: {qty[s]:.4f} ({pos_val[s]:,.2f}$)"for s in pos_val.index)or "Nothing yet")

                # Log the rebalance
                self._log(
                    f"{d.date()} | Rebalance\n"
                    f"  Buy   : {buy_usd:,.2f} $\n"
                    f"  Sell  : {sell_usd:,.2f} $\n"
                    f"  Fee   : {fee_paid:,.2f} $\n"
                    f"  Added : {need:,.2f} $, Invested total = {invested:,.2f} $\n"
                    f"  Cash  : {cash:,.2f} $ after tx\n"
                    f"  Traded: {traded_usd:,.2f} $ this rebalance  |  {traded_volume:,.2f} $ total\n"
                    f"  **Positions:** {pos_str}"
                )

            # Update the cost basis for the current position
            port_val = (qty * px_today).sum()
            equity = port_val + cash
            equity_curve.append({"datetime": d, "equity": equity})
            invested_curve.append({"datetime": d, "invested": invested})

        # Finalize the backtest
        final_unrealized = (qty * self.prices.iloc[-1]) - cost_basis_usd
        total_pnl = realized_pnl + final_unrealized

        self._log(
            f"Back‑test finished. Invested: {invested:,.2f} $ "
            f"Equity: {equity:,.2f} $ "
            f"Total fee: {total_fee:,.2f} $ "
            f"Turnover: {traded_volume:,.2f} $"
        )

        # Create the final equity and invested curves
        eq_s = pd.DataFrame(equity_curve).set_index("datetime")["equity"]
        inv_s = pd.DataFrame(invested_curve).set_index("datetime")["invested"]

        return BacktestResult(eq_s, inv_s, total_fee, total_pnl, traded_volume)

    @staticmethod
    def plot_equity(
        equity: pd.Series,
        invested: pd.Series,
        title_prefix: str = "",
        figsize: tuple[int, int] = (15, 8),
    ) -> None:
        run_max = equity.cummax()
        drawdown = (equity / run_max - 1).fillna(0)

        invested_usd = invested.iloc[-1]
        final_abs = equity.iloc[-1] - invested_usd
        final_pct = (equity.iloc[-1] / invested_usd - 1) * 100

        dd_flag = drawdown < 0
        days_dd = int(dd_flag.sum())
        grp = (dd_flag != dd_flag.shift()).cumsum()
        max_dd_len = int(dd_flag.groupby(grp).sum().max() if dd_flag.any() else 0)

        be_diff = equity - invested
        above_be = be_diff >= 0

        _, (ax_eq, ax_dd) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            dpi=300,
        )

        ax_eq.plot(equity, lw=1.2, color="skyblue", label="Portfolio")
        ax_eq.plot(invested, ls="--", color="green", label="Invested $")
        ax_eq.fill_between(
            equity.index, equity, invested, where=above_be, color="limegreen", alpha=0.25
        )
        ax_eq.fill_between(
            equity.index,
            equity,
            invested,
            where=~above_be,
            color="lightcoral",
            alpha=0.25,
        )

        title = (
            f"{title_prefix}\n"
            f"Final: {final_pct:.2f}% (USD {final_abs:,.0f}) • "
            f"Invested: USD {invested_usd:,.0f} • "
            f"DD days: {days_dd} • Longest DD: {max_dd_len}"
        )
        ax_eq.set_title(title)
        ax_eq.set_ylabel("USD")
        ax_eq.legend()
        ax_eq.grid(ls="--", alpha=0.4)

        ax_dd.plot(drawdown, color="crimson", lw=0.8)
        ax_dd.fill_between(drawdown.index, drawdown, 0, color="crimson", alpha=0.3)
        ax_dd.set_ylim(-1, 0)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("Date")
        ax_dd.grid(ls="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

        print(f"Final return      : {final_pct:.2f}% (USD {final_abs:,.0f})")
        print(f"Total invested    : USD {invested_usd:,.0f}")
        print(f"Time in drawdown  : {days_dd} trading days")
        print(f"Longest DD period : {max_dd_len} trading days")

    @staticmethod
    def report_top_movers(total_pnl: pd.Series, traded_volume: float, top_n: int = 5) -> None:
        """
        Report the top movers in the portfolio
        """
        top_gain = total_pnl.sort_values(ascending=False).head(top_n)
        top_loss = total_pnl.sort_values().head(top_n)

        print("\nTOP gainers, USD")
        print(top_gain.apply(lambda x: f"{x:,.2f}"))

        print("\nTOP losers, USD")
        print(top_loss.apply(lambda x: f"{x:,.2f}"))

        print(f"\nTraded volume, USD: {traded_volume:,.2f}")
