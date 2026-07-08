"""
Central configuration for the regime-shift asset allocation engine.

Keeping all the knobs in one place makes it easy to reproduce a run or try a
different setup without hunting through the code. Everything downstream imports
from here.
"""

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
# Three asset classes traded on India's NSE, plus the India VIX as a fear gauge.
#   ^NSEI         -> Nifty 50 index (equities, the "NSE data" we detect regimes on)
#   GOLDBEES.NS   -> Nippon India Gold ETF (gold)
#   LIQUIDBEES.NS -> Nippon India Liquid ETF (the safe / fixed-income leg)
ASSET_TICKERS = {
    "NIFTY": "^NSEI",
    "GOLD": "GOLDBEES.NS",
    "BONDS": "LIQUIDBEES.NS",
}

# The India VIX is a level (an index value), used only as a regime feature.
VIX_TICKER = "^INDIAVIX"

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# The equity ticker whose returns/volatility drive regime detection.
REGIME_ASSET = "NIFTY"

# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
MOMENTUM_WINDOWS = [21, 63]          # ~1 month and ~1 quarter
VOLATILITY_WINDOWS = [21, 63]        # rolling realized vol horizons
TRADING_DAYS = 252                   # for annualization

# ----------------------------------------------------------------------------
# HMM regime model
# ----------------------------------------------------------------------------
N_REGIMES = 3                        # Bull / Bear / Crisis
HMM_COVARIANCE_TYPE = "diag"         # fewer params, less overfitting (see guide 9.2)
HMM_N_ITER = 200
RANDOM_SEED = 42                     # reproducibility for the HMM's random init

# ----------------------------------------------------------------------------
# Walk-forward validation
# ----------------------------------------------------------------------------
MIN_TRAIN_SIZE = 756                 # first training window ~3 trading years
TEST_SIZE = 63                       # each out-of-sample block ~1 quarter

# The daily HMM regime signal is noisy, so we act on it at a realistic tactical
# cadence: rebalance on the first trading day of each month using the regime
# known at that point. Between rebalances the portfolio simply drifts. Daily
# rebalancing would rack up unrealistic turnover and let costs dominate.
REBALANCE_FREQUENCY = "monthly"      # "monthly" or "daily"

# ----------------------------------------------------------------------------
# Portfolio / backtest
# ----------------------------------------------------------------------------
TRANSACTION_COST_BPS = 10            # 10 basis points charged on turnover per rebalance

# Each regime uses a mean-variance objective with a different risk aversion:
#   Bull   -> low aversion  (return-seeking, leans into equities)
#   Bear   -> high aversion (defensive, tilts to gold/cash)
#   Crisis -> pure minimum variance (capital preservation)
BULL_RISK_AVERSION = 2.0
BEAR_RISK_AVERSION = 10.0
MAX_WEIGHT = 0.60                    # cap on any single asset (long-only, weights sum to 1)
MIN_REGIME_DAYS = 30                 # need this many training days to trust a regime's mean

# Static benchmark definitions (weights must sum to 1 over the three assets).
BENCHMARK_60_40 = {"NIFTY": 0.60, "GOLD": 0.00, "BONDS": 0.40}
BENCHMARK_EQUAL = {"NIFTY": 1 / 3, "GOLD": 1 / 3, "BONDS": 1 / 3}

# ----------------------------------------------------------------------------
# Output locations
# ----------------------------------------------------------------------------
FIGURE_DIR = "outputs"
RESULTS_DIR = "results"
