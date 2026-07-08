"""
Walk-forward validation and backtesting.

This is where the lookahead-bias discipline actually lives. We slide a sequence
of expanding train/test windows through time. Inside each fold we:

  1. compute z-score statistics on the TRAINING features only,
  2. re-fit the HMM on that scaled training data,
  3. decode regimes on the TEST window with a causal forward filter, so each
     test day's label depends only on data up to that day,
  4. estimate a covariance and regime-conditional expected returns from the
     TRAINING returns only,
  5. solve one optimal portfolio per regime and assign each test day the weights
     for its detected regime.

The resulting weight schedule is then fed through a backtest that lags the
weights by one day (you trade using what you knew at the previous close),
rebalances on a realistic schedule, and charges a transaction cost on turnover.
"""

import numpy as np
import pandas as pd

from . import config
from . import regime as regime_mod
from . import optimize
from . import features as feature_mod


def expanding_walk_forward_splits(n_obs, min_train_size, test_size):
    """Yield (train_idx, test_idx) position arrays for expanding-window folds.

    The training window always starts at 0 and grows; the test window is the
    fixed-size block immediately after it. Both slide forward until we run out
    of data.
    """
    splits = []
    start_test = min_train_size
    while start_test < n_obs:
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, min(start_test + test_size, n_obs))
        if len(test_idx) == 0:
            break
        splits.append((train_idx, test_idx))
        start_test += test_size
    return splits


def run_walk_forward(features, returns, verbose=True):
    """Run the full walk-forward regime detection + weight assignment.

    Parameters
    ----------
    features : DataFrame
        Raw (unscaled) feature matrix from ``features.build_features``.
    returns : DataFrame
        Daily log returns for the three assets.

    Returns
    -------
    regimes : Series
        Predicted regime label per out-of-sample day.
    target_weights : DataFrame
        Desired portfolio weights per out-of-sample day (columns = assets).
    """
    # Align returns to the feature dates (features lose leading rows to warmup).
    asset_returns = returns.loc[features.index]
    asset_names = list(config.ASSET_TICKERS.keys())
    asset_returns = asset_returns[asset_names]
    # Simple returns are the right thing for portfolio aggregation / Markowitz.
    simple_returns = np.expm1(asset_returns)

    feature_matrix = features.values
    vol_col = list(features.columns).index(feature_mod.volatility_feature_name())

    splits = expanding_walk_forward_splits(
        n_obs=len(features),
        min_train_size=config.MIN_TRAIN_SIZE,
        test_size=config.TEST_SIZE,
    )
    if verbose:
        print(f"Walk-forward: {len(splits)} folds "
              f"(min train {config.MIN_TRAIN_SIZE}d, test {config.TEST_SIZE}d each)")

    regime_pieces = []
    weight_pieces = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        # 1. Scale using training statistics only.
        scaler = regime_mod.StandardScaler()
        X_train = scaler.fit_transform(feature_matrix[train_idx])
        X_test = scaler.transform(feature_matrix[test_idx])

        # 2. Fit + label the HMM on the training window only.
        model, label_map = regime_mod.fit_and_label(X_train, vol_col)

        # 3. Causally filter states over [train | test] and split back out. The
        #    forward filter means each test day's regime depends only on data up
        #    to that day (no lookahead), while the long train history warms up
        #    the state beliefs so the labels stay persistent.
        all_states = regime_mod.filter_states(model, np.vstack([X_train, X_test]))
        train_states = all_states[:len(train_idx)]
        test_states = all_states[len(train_idx):]
        test_regimes = regime_mod.regimes_from_states(test_states, label_map)
        train_regimes = regime_mod.regimes_from_states(train_states, label_map)

        # 4. Covariance from the whole training window (stable); expected returns
        #    estimated *conditional on each regime* from the training data, so
        #    "Bull" reflects how assets behave in calm markets, etc.
        train_returns = simple_returns.iloc[train_idx]
        cov = train_returns.cov().values
        overall_mu = train_returns.mean().values

        regime_mu = {}
        for name in regime_mod.REGIME_NAMES:
            in_regime = train_regimes == name
            if in_regime.sum() >= config.MIN_REGIME_DAYS:
                regime_mu[name] = train_returns.values[in_regime].mean(axis=0)
            else:
                regime_mu[name] = overall_mu   # too few days -> fall back

        # 5. Solve one portfolio per regime, then map each test day to its regime.
        weights_by_regime = {
            name: optimize.regime_target_weights(name, regime_mu[name], cov)
            for name in regime_mod.REGIME_NAMES
        }
        test_dates = features.index[test_idx]
        fold_weights = np.vstack([weights_by_regime[r] for r in test_regimes])

        regime_pieces.append(pd.Series(test_regimes, index=test_dates))
        weight_pieces.append(pd.DataFrame(fold_weights, index=test_dates, columns=asset_names))

    regimes = pd.concat(regime_pieces)
    target_weights = pd.concat(weight_pieces)
    return regimes, target_weights


def rebalance_dates(index, frequency):
    """Pick the trading days on which the portfolio is actually rebalanced.

    "monthly" -> the first trading day of every calendar month.
    "daily"   -> every day.
    """
    if frequency == "daily":
        return set(index)
    if frequency == "monthly":
        as_series = index.to_series()
        first_of_month = as_series.groupby([index.year, index.month]).head(1)
        return set(first_of_month)
    raise ValueError(f"Unknown rebalance frequency: {frequency}")


def simulate(target_weights, simple_returns, tc_bps, frequency):
    """Backtest a target-weight schedule with realistic rebalancing.

    On each rebalance day we trade the (drifted) current portfolio to that day's
    target weights and pay ``tc_bps`` basis points on the turnover. On every
    other day the portfolio just drifts with the market and we trade nothing.
    Targets are lagged one day so a position is only ever taken using
    information that was available beforehand (no lookahead).

    Returns three daily series: gross returns, net-of-cost returns, and the
    turnover incurred that day.
    """
    tc = tc_bps / 10000.0
    assets = list(target_weights.columns)

    # Lag: the target for day t is the regime weights known at the close of t-1.
    desired = target_weights.shift(1)
    rebal_days = rebalance_dates(target_weights.index, frequency)

    current = None    # current (drifting) portfolio weights; None until invested
    dates, gross_returns, net_returns, turnovers = [], [], [], []

    for date in target_weights.index:
        day_returns = simple_returns.loc[date, assets].values
        target = desired.loc[date].values
        turnover = 0.0

        # Rebalance only on scheduled days once a valid target exists.
        if date in rebal_days and not np.isnan(target).any():
            previous = np.zeros(len(assets)) if current is None else current
            turnover = np.abs(target - previous).sum()
            current = target.copy()

        if current is None:
            continue   # not invested yet (before the first rebalance)

        gross = float(current @ day_returns)
        cost = tc * turnover
        dates.append(date)
        gross_returns.append(gross)
        net_returns.append(gross - cost)
        turnovers.append(turnover)

        # Let the position drift with today's returns into tomorrow.
        grown = current * (1 + day_returns)
        total = grown.sum()
        current = grown / total if total > 0 else current

    return (
        pd.Series(gross_returns, index=dates, name="gross"),
        pd.Series(net_returns, index=dates, name="net"),
        pd.Series(turnovers, index=dates, name="turnover"),
    )


def static_weight_schedule(benchmark_dict, index):
    """Build a constant-weight schedule (a benchmark) over the given dates."""
    asset_names = list(config.ASSET_TICKERS.keys())
    row = np.array([benchmark_dict[a] for a in asset_names])
    data = np.tile(row, (len(index), 1))
    return pd.DataFrame(data, index=index, columns=asset_names)
