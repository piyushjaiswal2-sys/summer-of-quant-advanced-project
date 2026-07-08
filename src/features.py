"""
Feature engineering for regime detection.

We describe the state of the market with two families of signals computed on the
equity index (Nifty), plus the India VIX:

  * momentum  -> is the market trending up or down? (rolling total return)
  * volatility -> how much is it swinging? (rolling std of returns, annualized)

These are the raw, *unscaled* features. All z-scoring is done later inside the
walk-forward loop using training-window statistics only, so nothing here peeks
into the future.
"""

import numpy as np
import pandas as pd

from . import config


def build_features(returns, vix):
    """Build the raw feature matrix fed to the HMM.

    Parameters
    ----------
    returns : DataFrame
        Daily log returns for the assets (must contain the regime asset column).
    vix : Series
        India VIX level aligned to ``returns``.

    Returns
    -------
    DataFrame
        One row per trading day, columns = the engineered features. Leading rows
        with NaNs (from the rolling windows warming up) are dropped.
    """
    equity_returns = returns[config.REGIME_ASSET]
    # Reconstruct a price level from the log returns so momentum is a clean
    # cumulative return over each window.
    equity_price = np.exp(equity_returns.cumsum())

    features = pd.DataFrame(index=returns.index)

    # Momentum: total return over the last N days at a couple of horizons.
    for window in config.MOMENTUM_WINDOWS:
        features[f"mom_{window}"] = equity_price.pct_change(window)

    # Volatility: rolling standard deviation of returns, annualized.
    for window in config.VOLATILITY_WINDOWS:
        features[f"vol_{window}"] = equity_returns.rolling(window).std() * np.sqrt(config.TRADING_DAYS)

    # The VIX itself is a forward-looking fear gauge and a strong regime signal.
    features["vix"] = vix

    features = features.dropna()
    return features


def volatility_feature_name():
    """Name of the volatility feature used to label HMM states (highest = Crisis).

    We use the shortest volatility window because it reacts fastest to stress.
    """
    return f"vol_{config.VOLATILITY_WINDOWS[0]}"
