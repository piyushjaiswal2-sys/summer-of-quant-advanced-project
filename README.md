# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

A quant strategy that figures out whether the market is calm, falling, or in a
full-blown crisis, and reshuffles a portfolio between **equities, gold, and a
safe (liquid debt) asset** to match. Instead of a fixed split like 60/40 that
never adapts, it detects hidden market *regimes* with a Hidden Markov Model and
picks portfolio weights that suit whatever regime we're currently in.

The whole thing is validated **walk-forward** so that nothing in the results
depends on information the strategy couldn't have had at the time — avoiding
lookahead bias, which is the single easiest way to build a backtest that looks
amazing and then loses money live.

Markets used are Indian (NSE): **Nifty 50** for equities, **GOLDBEES** for gold,
**LIQUIDBEES** as the safe/fixed-income leg, and the **India VIX** as a fear
gauge feature.

## What it does, end to end

```
data  →  features  →  regime detection (HMM)  →  optimization (CVXPY)  →  walk-forward backtest  →  results
```

1. **Data** — pull daily adjusted prices from Yahoo Finance, clean obvious bad
   ticks, convert to log returns, align everything on one calendar.
2. **Features** — momentum and rolling volatility of the equity index at a
   couple of horizons, plus the VIX level.
3. **Regime detection** — a 3-state Gaussian HMM (`hmmlearn`) infers a
   Bull/Bear/Crisis label for each day. States are named automatically by their
   average volatility (no manual labelling of any day).
4. **Optimization** — for each regime, solve for portfolio weights with `cvxpy`
   using a different objective (aggressive in Bull, defensive in Bear, minimum
   variance in Crisis).
5. **Walk-forward backtest** — re-fit the HMM inside each training window, decode
   regimes causally on the test window, and trade a monthly-rebalanced portfolio
   with transaction costs. Compare against static 60/40 and equal-weight.

## Results

Out-of-sample (walk-forward) performance, net of 10 bps transaction costs, on
data from 2010 to end-2024:

| Strategy       | Ann. Return | Ann. Vol | Sharpe | Sortino | Max Drawdown | Calmar | Ann. Turnover |
|----------------|------------:|---------:|-------:|--------:|-------------:|-------:|--------------:|
| **Dynamic (net)** |      6.1% |     6.5% |   0.94 |    1.23 |   **-13.4%** |   0.45 |          3.89 |
| Static 60/40   |        9.6% |     9.8% |   0.98 |    1.23 |       -23.7% |   0.41 |          0.29 |
| Equal weight   |        9.0% |     6.6% |   1.33 |    1.80 |       -15.2% |   0.59 |          0.36 |

Effect of transaction costs on the dynamic strategy: annual return goes from
6.5% (gross) to 6.1% (net) — monthly rebalancing keeps turnover low enough that
costs don't dominate.

**How to read this.** The dynamic strategy's headline is **downside
protection**: its worst drawdown (-13.4%) is far shallower than the 60/40
portfolio's (-23.7%), and it beats 60/40 on Calmar at a comparable Sharpe and
much lower volatility. It does that by de-risking into the safe asset when the
HMM flags a crisis — you can see the shallow dip through the March-2020 COVID
crash in `outputs/07_equity_curves.png`. It does **not** beat the equal-weight
portfolio here, largely because gold had an exceptional decade in India and a
constant 1/3 gold allocation was hard to beat; I've reported that honestly
rather than tune parameters until the strategy "wins". (Numbers come from live
Yahoo Finance data over a fixed date range, so re-running reproduces them.)

The HMM's transition matrix (rows sum to 1) confirms regimes are sticky, which
is exactly what we'd expect — the market rarely jumps straight from calm to
crisis in a single day:

|        | Bull | Bear | Crisis |
|--------|-----:|-----:|-------:|
| Bull   | 0.99 | 0.01 |   0.00 |
| Bear   | 0.01 | 0.98 |   0.00 |
| Crisis | 0.00 | 0.02 |   0.98 |

## Key design decisions

**Why 3 regimes?** The project asks for Bull / Bear / Crisis, and three is the
smallest number that captures the qualitatively different states we care about:
calmly rising, steadily falling, and violent high-volatility stress. Two states
can't separate "falling" from "crashing", and more than three tends to split
the data into economically meaningless clusters that overfit noise.

**Why these features?** Price level alone tells you nothing about the regime —
the same index level can be reached by a calm grind up or a violent
crash-and-recover. What discriminates regimes is *direction* (momentum) and
*uncertainty* (volatility), so we feed the HMM momentum and rolling volatility
at 1-month and 1-quarter horizons plus the India VIX, which is a direct,
forward-looking measure of expected volatility.

**Why LIQUIDBEES as the "bond"?** Indian government-bond ETFs on Yahoo Finance
have short or unreliable histories (implausible daily jumps, stale prices).
LIQUIDBEES is a liquid debt ETF with a clean 15-year history and behaves as the
portfolio's safe/cash leg — which is exactly how many Indian investors use it.
Its returns are steady and its volatility is tiny, making it the natural
flight-to-safety asset.

**Why mean-variance instead of a literal "maximize Sharpe" in Bull?** Because
the safe asset is essentially cash: a tiny volatility with a steady positive
return gives it a huge Sharpe ratio, so a pure Sharpe-maximizer parks almost the
entire book in cash in *every* regime and never takes any equity risk — which
defeats the whole point. Instead each regime uses a mean-variance objective with
a **regime-specific risk aversion**, which is the same trade-off (return vs
risk) but lets the strategy actually lean into equities when it's safe to:

- **Bull** — low risk aversion → return-seeking, tilts to equities.
- **Bear** — high risk aversion → defensive, tilts to gold and the safe asset.
- **Crisis** — pure minimum variance → capital preservation.

Expected returns are estimated **conditional on each regime** (how assets behave
on the training days the HMM assigns to that regime), which is what makes "Bull"
favour equities and "Bear" favour gold. All portfolios are long-only, fully
invested, and capped at 60% in any single asset so no one asset dominates.

**Why monthly rebalancing?** The daily HMM signal is noisy and flips fairly
often. Trading on every flip racks up enormous turnover and lets transaction
costs quietly destroy returns (a genuine trap the project warns about). Acting
once a month — a realistic tactical cadence — keeps turnover sane while still
capturing the regime shifts that matter. You can switch to daily via
`REBALANCE_FREQUENCY` in `src/config.py` and watch the costs bite.

## How lookahead bias is avoided

This is the part the project cares about most. At every point in time `t`, every
number used to make a decision for `t` is computable from data dated `≤ t`:

- The HMM is **re-fit inside each walk-forward fold** on training data only —
  never once on the full sample and then "predicted" backwards.
- Feature **z-scoring uses training-window statistics only** (`StandardScaler`
  is `fit` on train, then `transform` applied to test).
- Test-day regimes are decoded with a **causal forward filter**
  (`regime.filter_states`), not Viterbi or the smoothing posterior — both of
  those use the entire sequence, including future days, to label any given day.
- Expected returns and covariance for the optimizer come from the **training
  window only**.
- Target weights are **lagged one day** in the backtest, so a position is only
  ever taken using information available beforehand.

## Repository layout

```
regime-shift-engine/
├── main.py                     # runs the whole pipeline, saves figures + tables
├── requirements.txt
├── README.md
├── src/
│   ├── config.py               # all tunable parameters in one place
│   ├── data.py                 # download + clean + align prices/returns/VIX
│   ├── features.py             # momentum + volatility feature engineering
│   ├── regime.py               # HMM fit, volatility-based labelling, causal filter
│   ├── optimize.py             # per-regime CVXPY portfolio objectives
│   ├── backtest.py             # walk-forward splits, simulation, benchmarks
│   ├── metrics.py              # Sharpe / Sortino / drawdown / Calmar / turnover
│   └── plots.py                # regime overlays, transition matrix, equity curves
├── notebooks/
│   ├── regime_shift_engine.ipynb   # full pipeline top-to-bottom, with narrative
│   └── _build_notebook.py          # regenerates the notebook from the src package
├── outputs/                    # generated figures (regime charts, equity curves, ...)
└── results/                    # generated CSV tables (performance, costs, transitions)
```

## Setup

Requires Python 3.9+.

```bash
git clone <your-repo-url>
cd regime-shift-engine
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run the full pipeline as a script (downloads data, fits everything, writes all
figures to `outputs/` and tables to `results/`, and prints a summary):

```bash
python main.py
```

Or step through the annotated notebook:

```bash
jupyter notebook notebooks/regime_shift_engine.ipynb
```

The notebook imports the same `src` package `main.py` uses, so the two can't
drift apart. To regenerate the notebook file itself after editing the pipeline,
run `python notebooks/_build_notebook.py`.

## Reproducing the results

Results are deterministic given the data:

- the HMM uses a fixed random seed (`RANDOM_SEED` in `src/config.py`),
- the data range is fixed (`START_DATE` / `END_DATE`), and
- Yahoo Finance serves the same adjusted history for that range.

So a fresh `python main.py` on any machine reproduces the tables above (up to
any later Yahoo data revisions). Every knob — tickers, feature windows, number
of regimes, risk aversions, transaction cost, rebalance frequency, walk-forward
window sizes — lives in `src/config.py`, so experiments are one edit away.

## Tech stack

Python · NumPy · Pandas · SciPy · Matplotlib · yfinance · hmmlearn · CVXPY
