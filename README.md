# Trading Portfolio Rebalancer

This project implements a modular and extensible trading portfolio rebalancer designed for backtesting simple quantitative strategies on cryptocurrency market data. It uses historical OHLCV data from Binance and simulates rebalancing a portfolio of selected assets over time, accounting for trading fees and capital allocation dynamics.

---

## Contents

- [Overview](#overview)
- [Implemented Strategies](#implemented-strategies)
  - [TopNEqualDollarStrategy](#topnequaldollarstrategy)
  - [DynamicEWMAStrategy](#dynamicewmastrategy)
  - [Custom Strategies](#custom-strategies)
- [How It Works](#how-it-works)
- [Using It in a Live Scenario](#using-it-in-a-live-scenario)
- [Sample Output](#sample-output)
- [Installation](#installation)
- [ MOST IMPORTANT! Example and Results](#example-and-results)

---

## Overview

The rebalancer builds a trading portfolio out of N strategies with the best performance and adjusts it periodically (e.g., weekly). You can evaluate the system using historical market data, simulate buys and sells with trading fees, and visualize the results.

- Assets are selected from the top-50 most liquid USDT pairs.
- Backtest logic runs weekly rebalancing based on a rolling lookback window.
- Trades are evaluated in USD and plotted over time.
- Strategy selection is modular and customizable.

---

## Implemented Strategies

### TopNEqualDollarStrategy

This is a simple buy-and-hold style strategy. Every rebalance interval, it:

- Looks back over a fixed window of historical prices (e.g., 30 days).
- Computes performance for each asset:  
  `performance = (final_price / initial_price) - 1`
- Selects the top N assets with the highest returns.
- Allocates a fixed USD amount (`usd_per_strat`) to each of the selected assets equally.

This strategy does not make predictions; it purely ranks past performance and reallocates accordingly.

### DynamicEWMAStrategy

This strategy picks assets that have deviated the most from their own exponential moving average (EWMA), under the assumption that extreme deviations may be followed by reversions.

- Computes the EWMA over a fixed span (e.g., 25 days).
- Calculates the absolute percentage deviation between the current price and EWMA.
- Selects the N assets with the largest deviations.
- Allocates a fixed USD amount to each of them.

This strategy does improve PnL in our specific test case, but it primarily serves to demonstrate how a user can define custom strategy logic by overriding `compute_target_usd()` and implementing alternative asset selection logic.

---

## Custom Strategies

You can create your own strategy by subclassing `BaseStrategy`. The only requirement is to implement the method:

```python
def compute_target_usd(
    self, window: pd.DataFrame, px_today: pd.Series
) -> pd.Series:
```

Where:

-   `window` is a DataFrame of historical closing prices (assets × time).
    
-   `px_today` is a Series of today's prices (latest close).
    
-   The return value must be a Series that maps asset symbols to target USD allocations. Assets not selected should have a value of 0.
    
Example:

```python
pd.Series({
    "BTC/USDT": 1000.0,
    "ETH/USDT": 1000.0,
    ...
})

```

## How It Works

1.  Load OHLCV data using `CexDataLoader`, selecting top-volume USDT pairs.
    
2.  Initialize a strategy.
    
3.  Run the `Rebalancer` with the loaded data, rebalancing frequency, and lookback period.
    
4.  Simulate portfolio evolution by computing changes in allocation, applying fees, and tracking invested capital.
    
5.  Generate a plot of equity vs invested capital and a drawdown chart.
    
6.  Display performance summary including top gainers/losers.
    
---

## Using It in a Live Scenario

In a live trading environment:

-   Replace historical price fetching with live OHLCV updates.
    
-   Use the same strategy logic to compute target allocations at scheduled intervals.
    
-   Compare the target allocation against the current portfolio and place the necessary trades.
    
-   Record fees, PnL, and rebalancing activity.
    
-   You can also simulate partial fills, slippage, and execution latency for realism.
    

This architecture supports plug-and-play strategies and could be integrated into a trading bot framework with minimal modification.

---

## Sample Output

The script produces:

-   A plot of portfolio value vs total invested capital.
    
-   A drawdown graph.
    
-   Console logs of rebalancing decisions, fees, and trading volume.
    
-   Printed summaries of:
    
    -   Final returns
        
    -   Drawdown duration
        
    -   Traded volume
        
    -   Top-performing and worst-performing assets

---

## Installation

To set up the project:

1. Clone the repository.
2. Create a virtual environment:
3. Activate the virtual environment:
- On Unix/macOS:
  ```
  source venv/bin/activate
  ```
- On Windows:
  ```
  venv\Scripts\activate
  ```
4. Install the required packages:
 ```
python3.12 -m pip install -r requirements.txt
  ```

---

## Example and Results

In this example, we applied the rebalancer to historical Binance data starting from May 2024, using two different strategies:

1.  **TopNEqualDollarStrategy** (Buy & Hold of best performers based on past 30-day return)
    
2.  **DynamicEWMAStrategy** (Top N based on price deviation from EWMA)
    

Both strategies were rebalanced weekly using a 30-day lookback window, with a fixed $1,000 allocation per selected asset and 0.1% trading fee.

### Buy & Hold Strategy

-   Final return: **+68.81%**
    
-   Total invested: **$7,459**
    
-   Top gainers: **XRP/USDT**, **DOGE/USDT**, **CRV/USDT**
    
-   Top losers: **PEOPLE/USDT**, **1000SATS/USDT**
    
-   Time in drawdown: **292 days**, longest drawdown period: **160 days**

![image](https://github.com/user-attachments/assets/b98d1f08-9de4-4e72-a241-b60c65c1e95b)
    

This strategy performed steadily, especially benefiting from the upward momentum of some large-cap tokens.

### EWMA Deviation Strategy

### EWMA Deviation Strategy

-   Final return: **+164.17%**
-   Total invested: **$7,047**
-   Top gainers: **ALPACA/USDT**, **HBAR/USDT**, **CRV/USDT**
-   Top losers: **FLOKI/USDT**, **1000SATS/USDT**, **BONK/USDT**
-   Time in drawdown: **294 days**, longest drawdown period: **135 days**

![image](https://github.com/user-attachments/assets/36b8d2f8-5067-4256-89e7-49719afec5e5)


By decreasing the EWMA span to 15 and rebalancing every 5 days, this configuration delivered the strongest performance among tested variants. It captured short-term reversion patterns effectively, with high returns driven by top movers like ALPACA and CRV. Despite increased turnover and exposure to volatile tokens, it maintained controlled drawdowns and demonstrated the benefits of fine-tuned, dynamic allocation logic.
### Interpretation

The EWMA strategy demonstrates that customizing logic to consider short-term market anomalies (e.g., deviations from EWMA) can meaningfully influence PnL, asset exposure, and risk profile. While it did not outperform buy-and-hold in this specific instance, it shows how different allocation mechanisms react to market structure.

This example highlights the **flexibility** of the rebalancer framework and how modifying `compute_target_usd()` enables testing various portfolio construction logics without modifying the core engine.

