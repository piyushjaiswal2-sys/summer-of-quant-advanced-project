"""
Regime-Shift: Macro-Aware Tactical Asset Allocation Engine
==========================================================

Runs the whole pipeline top to bottom:

    data -> features -> regime detection -> optimization -> backtest -> results

It saves every figure to ``outputs/`` and the performance tables to
``results/``, and prints a summary to the console. Just run:

    python main.py

Everything is reproducible: the HMM uses a fixed random seed and the data is
pulled live from Yahoo Finance for a fixed date range.
"""

import os

import numpy as np
import pandas as pd

from src import config
from src import data as data_mod
from src import features as feature_mod
from src import regime as regime_mod
from src import backtest as backtest_mod
from src import metrics as metrics_mod
from src import plots as plot_mod


def ensure_dirs():
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def fig(name):
    return os.path.join(config.FIGURE_DIR, name)


def main():
    ensure_dirs()
    np.random.seed(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n=== 1. Loading data ===")
    prices, returns, vix = data_mod.load_dataset()
    plot_mod.plot_asset_returns(prices, returns, save_path=fig("01_asset_returns.png"))

    # ------------------------------------------------------------------
    # 2. Features
    # ------------------------------------------------------------------
    print("\n=== 2. Engineering features ===")
    features = feature_mod.build_features(returns, vix)
    print("Features:", list(features.columns))
    print(f"{len(features)} usable days after warmup "
          f"({features.index[0].date()} -> {features.index[-1].date()})")
    plot_mod.plot_volatility_check(features, save_path=fig("02_volatility_check.png"))

    # ------------------------------------------------------------------
    # 3. Full-sample HMM (for the regime picture + transition matrix)
    #    NOTE: this in-sample fit is for VISUALIZATION only. The honest,
    #    leak-free evaluation is the walk-forward backtest further down.
    # ------------------------------------------------------------------
    print("\n=== 3. Fitting HMM on full sample (visualization only) ===")
    scaler = regime_mod.StandardScaler()
    X_all = scaler.fit_transform(features.values)
    vol_col = list(features.columns).index(feature_mod.volatility_feature_name())
    full_model, full_label_map = regime_mod.fit_and_label(X_all, vol_col)
    full_regimes = pd.Series(
        regime_mod.predict_regimes(full_model, full_label_map, X_all),
        index=features.index, name="regime",
    )

    # Re-order the raw transition matrix into Bull/Bear/Crisis order.
    order = [None] * config.N_REGIMES
    for state, name in full_label_map.items():
        order[regime_mod.REGIME_NAMES.index(name)] = state
    ordered_transmat = full_model.transmat_[np.ix_(order, order)]
    print("Transition matrix (rows/cols = Bull, Bear, Crisis):")
    print(pd.DataFrame(ordered_transmat, index=regime_mod.REGIME_NAMES,
                       columns=regime_mod.REGIME_NAMES).round(3))
    print("\nRegime day counts:")
    print(full_regimes.value_counts())

    plot_mod.plot_regime_overlay(
        prices, full_regimes,
        title="HMM-inferred regimes on Nifty 50 (full-sample fit, in-sample)",
        save_path=fig("03_regimes_full_sample.png"))
    plot_mod.plot_transition_matrix(ordered_transmat, save_path=fig("04_transition_matrix.png"))

    # ------------------------------------------------------------------
    # 4. Walk-forward validation (the real evaluation, no lookahead)
    # ------------------------------------------------------------------
    print("\n=== 4. Walk-forward regime detection + optimization ===")
    splits = backtest_mod.expanding_walk_forward_splits(
        len(features), config.MIN_TRAIN_SIZE, config.TEST_SIZE)
    plot_mod.plot_walk_forward_folds(splits, features.index, save_path=fig("05_walk_forward_folds.png"))

    oos_regimes, target_weights = backtest_mod.run_walk_forward(features, returns)
    print(f"Out-of-sample period: {oos_regimes.index[0].date()} -> "
          f"{oos_regimes.index[-1].date()} ({len(oos_regimes)} days)")
    print("Out-of-sample regime counts:")
    print(oos_regimes.value_counts())

    plot_mod.plot_regime_overlay(
        prices, oos_regimes,
        title="Walk-forward (out-of-sample) regimes on Nifty 50",
        save_path=fig("06_regimes_walk_forward.png"))

    # ------------------------------------------------------------------
    # 5. Backtest: dynamic strategy vs static benchmarks
    # ------------------------------------------------------------------
    print("\n=== 5. Backtesting ===")
    simple_returns = np.expm1(returns[list(config.ASSET_TICKERS.keys())])
    oos_index = target_weights.index

    freq = config.REBALANCE_FREQUENCY

    # Dynamic regime-switching strategy (with and without costs).
    dyn_gross, dyn_net, dyn_turn = backtest_mod.simulate(
        target_weights, simple_returns, config.TRANSACTION_COST_BPS, freq)
    dyn_gross_only, _, _ = backtest_mod.simulate(target_weights, simple_returns, 0, freq)

    # Static benchmarks over the same out-of-sample window (rebalanced on the
    # same schedule so the comparison is apples-to-apples).
    bench_6040_w = backtest_mod.static_weight_schedule(config.BENCHMARK_60_40, oos_index)
    bench_eq_w = backtest_mod.static_weight_schedule(config.BENCHMARK_EQUAL, oos_index)
    _, bench_6040_net, bench_6040_turn = backtest_mod.simulate(
        bench_6040_w, simple_returns, config.TRANSACTION_COST_BPS, freq)
    _, bench_eq_net, bench_eq_turn = backtest_mod.simulate(
        bench_eq_w, simple_returns, config.TRANSACTION_COST_BPS, freq)

    # ------------------------------------------------------------------
    # 6. Results
    # ------------------------------------------------------------------
    print("\n=== 6. Results ===")

    net_results = {
        "Dynamic (net)": (dyn_net, dyn_turn),
        "Static 60/40": (bench_6040_net, bench_6040_turn),
        "Equal weight": (bench_eq_net, bench_eq_turn),
    }
    net_table = metrics_mod.summary_table(net_results)

    cost_compare = metrics_mod.summary_table({
        "Dynamic (gross)": (dyn_gross_only, dyn_turn),
        "Dynamic (net of costs)": (dyn_net, dyn_turn),
    })

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

    print("\nPerformance summary (net of transaction costs):")
    print(net_table.round(3))
    print("\nEffect of transaction costs on the dynamic strategy:")
    print(cost_compare.round(3))

    # Save tables to results/.
    net_table.round(4).to_csv(os.path.join(config.RESULTS_DIR, "performance_summary.csv"))
    cost_compare.round(4).to_csv(os.path.join(config.RESULTS_DIR, "cost_impact.csv"))
    pd.DataFrame(ordered_transmat, index=regime_mod.REGIME_NAMES,
                 columns=regime_mod.REGIME_NAMES).round(4).to_csv(
        os.path.join(config.RESULTS_DIR, "transition_matrix.csv"))

    # Equity curve comparison plot.
    plot_mod.plot_equity_curves({
        "Dynamic (net)": dyn_net,
        "Static 60/40": bench_6040_net,
        "Equal weight": bench_eq_net,
    }, save_path=fig("07_equity_curves.png"))

    print("\nDone. Figures saved to ./outputs, tables saved to ./results.")


if __name__ == "__main__":
    main()
