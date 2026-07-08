"""
Convex portfolio optimization with CVXPY.

Each regime gets its own objective, which is the whole point of a regime-aware
strategy. We use a mean-variance objective whose risk aversion changes with the
regime:

  * Bull   -> low risk aversion  -> return-seeking, leans into equities
  * Bear   -> high risk aversion -> defensive, tilts to gold / cash
  * Crisis -> minimum variance   -> flight to safety

Why mean-variance instead of a literal "maximize Sharpe" in Bull? Our safe asset
(LIQUIDBEES) is essentially cash: tiny volatility and a steady positive return,
so its risk-adjusted return dominates everything else. A pure Sharpe-maximizer
would therefore dump almost the entire book into cash in every regime, which
defeats the purpose. Mean-variance with a regime-dependent risk appetite is the
same idea (trade return against risk) but lets us actually take equity risk when
the market is calm. See the README for the full reasoning.

All portfolios are long-only, weights sum to 1, and no single asset may exceed
``MAX_WEIGHT``. Expected returns (mu) and covariance (Sigma) are always
estimated from the training window only, so the optimizer never sees the future.
"""

import numpy as np
import cvxpy as cp

from . import config


def _base_constraints(w):
    """Long-only, fully invested, capped at MAX_WEIGHT per asset."""
    return [cp.sum(w) == 1, w >= 0, w <= config.MAX_WEIGHT]


def min_variance_weights(cov):
    """Long-only minimum-variance portfolio: min wᵀ Σ w."""
    n = cov.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    cp.Problem(objective, _base_constraints(w)).solve()
    return _clean_weights(w.value, n)


def mean_variance_weights(mu, cov, risk_aversion):
    """Long-only mean-variance portfolio: max muᵀw - risk_aversion * wᵀΣw.

    A low ``risk_aversion`` chases return (Bull); a high one prioritizes low
    variance (Bear).
    """
    n = cov.shape[0]
    w = cp.Variable(n)
    expected_return = mu @ w
    variance = cp.quad_form(w, cp.psd_wrap(cov))
    objective = cp.Maximize(expected_return - risk_aversion * variance)
    cp.Problem(objective, _base_constraints(w)).solve()
    return _clean_weights(w.value, n)


def regime_target_weights(regime_name, mu, cov):
    """Dispatch to the right objective for the detected regime."""
    if regime_name == "Bull":
        return mean_variance_weights(mu, cov, config.BULL_RISK_AVERSION)
    elif regime_name == "Bear":
        return mean_variance_weights(mu, cov, config.BEAR_RISK_AVERSION)
    elif regime_name == "Crisis":
        return min_variance_weights(cov)
    else:
        raise ValueError(f"Unknown regime: {regime_name}")


def _clean_weights(weights, n):
    """Tidy up solver output: handle failures and tiny negative rounding noise."""
    if weights is None:
        # Solver failed -> fall back to equal weight rather than crashing.
        return np.ones(n) / n
    weights = np.clip(weights, 0, None)   # kill tiny negatives from the solver
    total = weights.sum()
    if total <= 0:
        return np.ones(n) / n
    return weights / total
