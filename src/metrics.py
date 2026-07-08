"""
Performance metrics for the backtest.

All ratios are annualized from daily returns. These are the standard yardsticks
the project asks for: Sharpe, Sortino, max drawdown, Calmar, plus average
annual turnover as a cost/activity measure.
"""

import numpy as np
import pandas as pd

from . import config


def annualized_return(daily_returns):
    """Geometric annualized return from a series of daily simple returns."""
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan
    total_growth = (1 + daily_returns).prod()
    years = len(daily_returns) / config.TRADING_DAYS
    return total_growth ** (1 / years) - 1


def annualized_volatility(daily_returns):
    return daily_returns.dropna().std() * np.sqrt(config.TRADING_DAYS)


def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """Annualized Sharpe ratio (excess return per unit of total volatility)."""
    excess = daily_returns.dropna() - risk_free_rate / config.TRADING_DAYS
    if excess.std() == 0:
        return np.nan
    return np.sqrt(config.TRADING_DAYS) * excess.mean() / excess.std()


def sortino_ratio(daily_returns, risk_free_rate=0.0):
    """Like Sharpe but only penalizes downside (below-target) volatility."""
    excess = daily_returns.dropna() - risk_free_rate / config.TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return np.sqrt(config.TRADING_DAYS) * excess.mean() / downside_std


def max_drawdown(daily_returns):
    """Worst peak-to-trough decline of the compounded equity curve (negative)."""
    equity = (1 + daily_returns.dropna()).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return drawdown.min()


def calmar_ratio(daily_returns):
    """Annualized return divided by the absolute max drawdown."""
    mdd = max_drawdown(daily_returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(daily_returns) / abs(mdd)


def average_annual_turnover(turnover_series):
    """Mean daily turnover scaled to an annual figure (rough activity gauge)."""
    if turnover_series is None or len(turnover_series) == 0:
        return 0.0
    return turnover_series.mean() * config.TRADING_DAYS


def equity_curve(daily_returns):
    """Growth of 1 unit of capital over time."""
    return (1 + daily_returns.fillna(0)).cumprod()


def drawdown_series(daily_returns):
    equity = equity_curve(daily_returns)
    return equity / equity.cummax() - 1


def summarize(daily_returns, turnover_series=None):
    """Build a one-row dict of all headline metrics for a return stream."""
    return {
        "Ann. Return": annualized_return(daily_returns),
        "Ann. Volatility": annualized_volatility(daily_returns),
        "Sharpe": sharpe_ratio(daily_returns),
        "Sortino": sortino_ratio(daily_returns),
        "Max Drawdown": max_drawdown(daily_returns),
        "Calmar": calmar_ratio(daily_returns),
        "Ann. Turnover": average_annual_turnover(turnover_series),
    }


def summary_table(named_results):
    """Assemble a comparison DataFrame from {name: (returns, turnover)} items."""
    rows = {}
    for name, (returns, turnover) in named_results.items():
        rows[name] = summarize(returns, turnover)
    table = pd.DataFrame(rows).T
    return table[[
        "Ann. Return", "Ann. Volatility", "Sharpe", "Sortino",
        "Max Drawdown", "Calmar", "Ann. Turnover",
    ]]
