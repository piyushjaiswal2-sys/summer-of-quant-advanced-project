"""
Data loading and cleaning.

Pulls daily prices for our three assets plus the India VIX from Yahoo Finance,
cleans obvious bad ticks, and returns a single aligned table of daily log
returns together with the VIX level. Everything is aligned on one shared
DatetimeIndex via an inner join so there are no gaps to trip up the HMM later.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from . import config


def _download_close(tickers, start, end):
    """Download adjusted close prices for a list of tickers.

    auto_adjust=True keeps prices adjusted for dividends/splits, which matters a
    lot here: LIQUIDBEES pays daily distributions and using unadjusted prices
    would inject fake negative jumps into its return series.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    # With multiple tickers yfinance returns MultiIndex columns (field, ticker);
    # we only care about the Close field.
    close = raw["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close


def clean_prices(prices):
    """Repair obvious data errors using a rolling-median outlier filter.

    Yahoo's Indian gold-ETF history has a couple of days (around Dec 2019) where
    the price is off by a factor of ~100 and reverts the next day. Those single
    bad ticks would otherwise create absurd returns that dominate the HMM fit.

    Any close that is less than half or more than double the surrounding 11-day
    median is treated as an error, blanked out, and forward-filled. Real moves,
    even a 2020-style crash, never trip this test.
    """
    cleaned = prices.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        local_median = series.rolling(11, center=True, min_periods=3).median()
        ratio = series / local_median
        bad = (ratio < 0.5) | (ratio > 2.0)
        if bad.any():
            n_bad = int(bad.sum())
            bad_dates = [str(d.date()) for d in series.index[bad]]
            print(f"  cleaned {n_bad} bad tick(s) in {col}: {bad_dates}")
        cleaned[col] = series.mask(bad).ffill()
    return cleaned


def load_dataset(verbose=True):
    """Build the master dataset used by the rest of the pipeline.

    Returns
    -------
    prices : DataFrame
        Cleaned daily prices for the three assets (columns = NIFTY/GOLD/BONDS).
    returns : DataFrame
        Daily log returns for the three assets.
    vix : Series
        India VIX level, aligned to the same dates as ``returns``.
    """
    tickers = list(config.ASSET_TICKERS.values())
    ticker_to_name = {v: k for k, v in config.ASSET_TICKERS.items()}

    if verbose:
        print("Downloading asset prices:", ", ".join(tickers))
    asset_close = _download_close(tickers, config.START_DATE, config.END_DATE)
    asset_close = asset_close.rename(columns=ticker_to_name)
    asset_close = asset_close[list(config.ASSET_TICKERS.keys())]  # fixed column order

    if verbose:
        print("Downloading India VIX:", config.VIX_TICKER)
    vix_close = _download_close([config.VIX_TICKER], config.START_DATE, config.END_DATE)
    vix_close.columns = ["VIX"]

    # Clean bad ticks before doing anything else.
    if verbose:
        print("Cleaning prices...")
    asset_close = clean_prices(asset_close)

    # Log returns are additive across time and better behaved statistically.
    returns = np.log(asset_close).diff()

    # Align assets + VIX on a shared index; inner join drops any date missing
    # from one of the series (different holiday calendars, data gaps, etc.).
    merged = returns.join(vix_close["VIX"], how="inner").dropna()

    returns = merged[list(config.ASSET_TICKERS.keys())]
    vix = merged["VIX"]
    prices = asset_close.loc[returns.index]

    if verbose:
        print(f"Final aligned dataset: {len(returns)} trading days, "
              f"{returns.index[0].date()} -> {returns.index[-1].date()}")

    return prices, returns, vix
