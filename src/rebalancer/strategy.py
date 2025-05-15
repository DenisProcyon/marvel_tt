import numpy as np
import pandas as pd

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