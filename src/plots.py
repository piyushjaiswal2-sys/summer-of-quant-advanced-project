"""
Plotting helpers.

Every function either shows the figure (in a notebook) or, if a ``save_path`` is
given, writes it to disk. The three workhorse charts for this project are the
regime-overlay price chart, the transition-matrix heatmap, and the backtest
equity curve with its drawdown panel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from . import metrics
from . import regime as regime_mod

# Consistent colors for the three regimes throughout the project.
REGIME_COLORS = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_asset_returns(prices, returns, save_path=None):
    """Cumulative growth of each asset. Look at the data before doing anything clever."""
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    growth = (1 + np.expm1(returns)).cumprod()
    for col in growth.columns:
        ax[0].plot(growth.index, growth[col], label=col, lw=1.2)
    ax[0].set_title("Cumulative growth of 1 unit (each asset)")
    ax[0].set_ylabel("Growth")
    ax[0].legend()

    ax[1].plot(returns.index, returns[config.REGIME_ASSET], color="steelblue", lw=0.6)
    ax[1].set_title(f"Daily log returns ({config.REGIME_ASSET})")
    ax[1].set_ylabel("Return")
    _finish(fig, save_path)


def plot_volatility_check(features, save_path=None):
    """Sanity check: does the volatility feature spike in known crises?"""
    vol_name = f"vol_{config.VOLATILITY_WINDOWS[0]}"
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(features.index, features[vol_name], color="firebrick", lw=1)
    ax.set_title(f"{vol_name} (annualized), should spike around 2020 & 2022")
    ax.set_ylabel("Annualized volatility")
    # Shade a couple of well-known stress windows for reference.
    for start, end, label in [("2020-02-01", "2020-05-01", "COVID crash"),
                              ("2022-01-01", "2022-07-01", "2022 selloff")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.2)
    _finish(fig, save_path)


def plot_regime_overlay(prices, regimes, title, save_path=None):
    """Price line with regimes drawn as colored background bands."""
    price = prices[config.REGIME_ASSET].loc[regimes.index]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(price.index, price.values, color="black", lw=1, zorder=3)

    # Shade contiguous runs of the same regime.
    labels = regimes.values
    dates = regimes.index
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            ax.axvspan(dates[start], dates[min(i, len(labels) - 1)],
                       color=REGIME_COLORS[labels[start]], alpha=0.25, lw=0)
            start = i

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5)
               for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), loc="upper left")
    ax.set_title(title)
    ax.set_ylabel(f"{config.REGIME_ASSET} level")
    _finish(fig, save_path)


def plot_transition_matrix(transmat, save_path=None):
    """Heatmap of the HMM transition probabilities (rows sum to 1)."""
    labels = regime_mod.REGIME_NAMES
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(transmat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("To regime")
    ax.set_ylabel("From regime")
    ax.set_title("HMM transition probability matrix")
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center",
                    color="white" if transmat[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finish(fig, save_path)


def plot_walk_forward_folds(splits, index, save_path=None):
    """Visualize the expanding train/test fold structure."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for i, (train_idx, test_idx) in enumerate(splits):
        ax.barh(i, train_idx[-1] - train_idx[0], left=train_idx[0],
                color="steelblue", label="train" if i == 0 else "")
        ax.barh(i, test_idx[-1] - test_idx[0], left=test_idx[0],
                color="firebrick", label="test" if i == 0 else "")
    ax.set_yticks(range(len(splits)), [f"Fold {i+1}" for i in range(len(splits))])
    ax.set_xlabel("Observation index (time)")
    ax.set_title("Expanding-window walk-forward folds")
    ax.legend(loc="lower right")
    _finish(fig, save_path)


def plot_equity_curves(return_streams, save_path=None):
    """Equity curves (top) and drawdowns (bottom) for several strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for name, daily_returns in return_streams.items():
        equity = metrics.equity_curve(daily_returns)
        axes[0].plot(equity.index, equity.values, lw=1.3, label=name)
        dd = metrics.drawdown_series(daily_returns)
        axes[1].plot(dd.index, dd.values, lw=1, label=name)

    axes[0].set_title("Equity curves (net of transaction costs)")
    axes[0].set_ylabel("Growth of 1")
    axes[0].legend(loc="upper left")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    _finish(fig, save_path)
