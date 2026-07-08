"""
Generates regime_shift_engine.ipynb from a list of (type, source) cells.

Kept as a small helper so the notebook and the code stay in sync: the notebook
imports the same ``src`` package that ``main.py`` uses, so there is no
duplicated logic to drift out of date. Run:  python notebooks/_build_notebook.py
"""

import json
import os

CELLS = [
    ("md", """# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

This notebook runs the whole pipeline top to bottom:

**data -> features -> regime detection -> optimization -> backtest -> results**

We detect hidden market regimes (Bull / Bear / Crisis) on Indian markets with a
Hidden Markov Model, then let the detected regime drive a convex portfolio
optimization across equities (Nifty 50), gold, and a liquid debt ETF. Everything
is validated **walk-forward** so no result depends on information from the
future.

The heavy lifting lives in the `src/` package; this notebook just narrates the
story and calls into it, so the notebook and `main.py` can never disagree."""),

    ("code", """import sys, os
sys.path.append(os.path.abspath(".."))  # so `import src` works from notebooks/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src import data as data_mod
from src import features as feature_mod
from src import regime as regime_mod
from src import backtest as backtest_mod
from src import optimize as optimize_mod
from src import metrics as metrics_mod
from src import plots as plot_mod

plt.rcParams["figure.figsize"] = (12, 4.5)
np.random.seed(config.RANDOM_SEED)
print("Ready.")"""),

    ("md", """## 1. Data

We pull daily prices for three asset classes traded on the NSE, plus the India
VIX as a fear gauge. Prices are cleaned for obvious bad ticks and converted to
log returns, all aligned on one shared calendar."""),

    ("code", """prices, returns, vix = data_mod.load_dataset()
returns.tail()"""),

    ("code", """plot_mod.plot_asset_returns(prices, returns)"""),

    ("md", """## 2. Features

We describe the market's state with momentum and volatility of the equity index
at a couple of horizons, plus the VIX level. These are the *raw* features; all
z-scoring happens later, inside the walk-forward loop, using training data only.

As a sanity check, the volatility feature should spike around known stress
periods (the 2020 COVID crash and the 2022 selloff)."""),

    ("code", """features = feature_mod.build_features(returns, vix)
print("Features:", list(features.columns))
plot_mod.plot_volatility_check(features)"""),

    ("md", """## 3. Fitting the HMM (full-sample view)

First we fit one HMM on the whole sample just to *look* at the regimes and the
transition matrix. This is an in-sample picture for intuition only. The honest,
leak-free evaluation is the walk-forward backtest in Section 4.

The states come out unlabelled (0/1/2); we name them by average volatility:
calmest = Bull, most volatile = Crisis."""),

    ("code", """scaler = regime_mod.StandardScaler()
X_all = scaler.fit_transform(features.values)
vol_col = list(features.columns).index(feature_mod.volatility_feature_name())

full_model, full_label_map = regime_mod.fit_and_label(X_all, vol_col)
full_regimes = pd.Series(
    regime_mod.predict_regimes(full_model, full_label_map, X_all),
    index=features.index, name="regime")

# Put the transition matrix into Bull/Bear/Crisis order.
order = [None] * config.N_REGIMES
for state, name in full_label_map.items():
    order[regime_mod.REGIME_NAMES.index(name)] = state
ordered_transmat = full_model.transmat_[np.ix_(order, order)]

print("Regime day counts:")
print(full_regimes.value_counts(), "\\n")
print("Transition matrix (Bull/Bear/Crisis):")
pd.DataFrame(ordered_transmat, index=regime_mod.REGIME_NAMES,
             columns=regime_mod.REGIME_NAMES).round(3)"""),

    ("md", """The high diagonal of the transition matrix confirms regimes are
"sticky": once the market is in a state it tends to stay there, which is
exactly what we expect. Now overlay the regimes on the Nifty price."""),

    ("code", """plot_mod.plot_regime_overlay(
    prices, full_regimes,
    title="HMM-inferred regimes on Nifty 50 (full-sample, in-sample)")
plot_mod.plot_transition_matrix(ordered_transmat)"""),

    ("md", """## 4. Walk-forward validation

This is the part that keeps us honest. We slide expanding train/test windows
through time. In every fold we:

1. compute z-score stats on the **training** features only,
2. re-fit the HMM on that training window,
3. decode test-day regimes with a **causal forward filter** (each day uses only
   data up to that day (no lookahead),
4. estimate regime-conditional expected returns and a covariance from the
   **training** returns only,
5. solve one optimal portfolio per regime.

The result is an out-of-sample regime label and target weight for every day
after the first training window."""),

    ("code", """splits = backtest_mod.expanding_walk_forward_splits(
    len(features), config.MIN_TRAIN_SIZE, config.TEST_SIZE)
plot_mod.plot_walk_forward_folds(splits, features.index)"""),

    ("code", """oos_regimes, target_weights = backtest_mod.run_walk_forward(features, returns)
print("Out-of-sample regime counts:")
print(oos_regimes.value_counts())
plot_mod.plot_regime_overlay(
    prices, oos_regimes,
    title="Walk-forward (out-of-sample) regimes on Nifty 50")"""),

    ("md", """### What does each regime actually hold?

The average out-of-sample weights show the strategy behaving sensibly: heavy in
equities when calm (Bull), defensive and gold-tilted in Bear, and parked in the
safe asset during Crisis."""),

    ("code", """avg = target_weights.copy()
avg["regime"] = oos_regimes
avg.groupby("regime").mean().round(3)"""),

    ("md", """## 5. Backtest vs static benchmarks

We compare the dynamic regime-switching strategy against a static 60/40
(equity/bonds) portfolio and an equal-weight portfolio. All three rebalance on
the same monthly schedule, and we charge a transaction cost on turnover so the
comparison is fair. We report the dynamic strategy both gross and net of
costs."""),

    ("code", """freq = config.REBALANCE_FREQUENCY
simple_returns = np.expm1(returns[list(config.ASSET_TICKERS.keys())])
oos_index = target_weights.index

dyn_gross, dyn_net, dyn_turn = backtest_mod.simulate(
    target_weights, simple_returns, config.TRANSACTION_COST_BPS, freq)
dyn_gross_only, _, _ = backtest_mod.simulate(target_weights, simple_returns, 0, freq)

bench_6040_w = backtest_mod.static_weight_schedule(config.BENCHMARK_60_40, oos_index)
bench_eq_w = backtest_mod.static_weight_schedule(config.BENCHMARK_EQUAL, oos_index)
_, bench_6040_net, bench_6040_turn = backtest_mod.simulate(
    bench_6040_w, simple_returns, config.TRANSACTION_COST_BPS, freq)
_, bench_eq_net, bench_eq_turn = backtest_mod.simulate(
    bench_eq_w, simple_returns, config.TRANSACTION_COST_BPS, freq)
print("Backtest done.")"""),

    ("md", "### Performance summary (net of transaction costs)"),

    ("code", """net_table = metrics_mod.summary_table({
    "Dynamic (net)": (dyn_net, dyn_turn),
    "Static 60/40": (bench_6040_net, bench_6040_turn),
    "Equal weight": (bench_eq_net, bench_eq_turn),
})
net_table.round(3)"""),

    ("md", "### How much do transaction costs matter?"),

    ("code", """metrics_mod.summary_table({
    "Dynamic (gross)": (dyn_gross_only, dyn_turn),
    "Dynamic (net of costs)": (dyn_net, dyn_turn),
}).round(3)"""),

    ("md", "### Equity curves and drawdowns"),

    ("code", """plot_mod.plot_equity_curves({
    "Dynamic (net)": dyn_net,
    "Static 60/40": bench_6040_net,
    "Equal weight": bench_eq_net,
})"""),

    ("md", """## 6. Takeaways

- The HMM cleanly separates calm, choppy, and crisis markets with no hand
  labelling, and the Crisis band lines up with the 2020 COVID crash.
- Re-fitting inside each walk-forward fold, causal filtering, and train-only
  scaling mean every regime label depends only on the past.
- The dynamic strategy's headline result is **downside protection**: its worst
  drawdown is far shallower than the 60/40 portfolio's, at comparable
  risk-adjusted return, precisely because it de-risks into the safe asset when
  the HMM flags a crisis.
- Transaction costs are modest here (monthly rebalancing keeps turnover low),
  but the with/without-cost comparison shows why a noisier, daily-rebalanced
  version would bleed returns.

See the `README.md` for the reasoning behind the key design choices."""),
]


def build():
    cells = []
    for cell_type, source in CELLS:
        lines = source.split("\n")
        # nbformat wants every line except the last to end with a newline.
        source_lines = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if cell_type == "md":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source_lines,
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines,
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = os.path.join(os.path.dirname(__file__), "regime_shift_engine.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Wrote", out_path)


if __name__ == "__main__":
    build()
