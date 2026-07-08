"""
HMM-based regime classifier.

A Gaussian Hidden Markov Model treats the market regime as a hidden state that
we never observe directly; we only see the features it generates (momentum,
volatility, VIX). hmmlearn fits the emission distributions and the transition
matrix with Baum-Welch, and ``.predict()`` runs Viterbi to recover the most
likely sequence of hidden states.

The HMM hands back arbitrary state numbers (0, 1, 2). We give them meaning with
a simple, unsupervised heuristic: rank the states by their average volatility,
so the calmest state becomes Bull, the middle one Bear, and the most volatile
one Crisis. No day is ever labelled by hand.
"""

import numpy as np
from scipy.special import logsumexp
from hmmlearn import hmm

from . import config

# Canonical regime names ordered from calmest to most stressed.
REGIME_NAMES = ["Bull", "Bear", "Crisis"]


class StandardScaler:
    """Minimal z-score scaler that is fit on training data only.

    We roll our own (instead of sklearn's) to make the train-only discipline
    completely explicit: ``fit`` learns mean/std from the training window and
    ``transform`` applies those same numbers to any later data. The test set
    never sees its own statistics, which is exactly how lookahead bias is
    avoided during walk-forward validation.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Guard against a zero-variance column producing division by zero.
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def fit_hmm(X_train_scaled):
    """Fit a Gaussian HMM with ``N_REGIMES`` hidden states on scaled features."""
    model = hmm.GaussianHMM(
        n_components=config.N_REGIMES,
        covariance_type=config.HMM_COVARIANCE_TYPE,
        n_iter=config.HMM_N_ITER,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train_scaled)
    return model


def label_states_by_volatility(model, vol_column_index):
    """Map raw HMM state indices to Bull/Bear/Crisis by average volatility.

    Each hidden state has a learned mean vector (``model.means_``). We read off
    the volatility feature's mean for every state and sort: lowest average
    volatility -> Bull, highest -> Crisis. This is fully unsupervised - no day
    is labelled by hand. Returns a dict {state_index: regime_name}.
    """
    state_vol = model.means_[:, vol_column_index]
    ordered_states = np.argsort(state_vol)  # calmest first
    return {int(state): REGIME_NAMES[rank] for rank, state in enumerate(ordered_states)}


def fit_and_label(X_train_scaled, vol_column_index):
    """Convenience wrapper: fit the HMM and build its state->name mapping."""
    model = fit_hmm(X_train_scaled)
    label_map = label_states_by_volatility(model, vol_column_index)
    return model, label_map


def filter_states(model, X_scaled):
    """Causally decode the most likely state at each step (forward filtering).

    Unlike ``model.predict`` (Viterbi) and ``model.predict_proba`` (smoothing),
    which both use the *entire* sequence - including future observations - to
    label any given day, this runs only the forward pass. The state at time t is
    the argmax of the filtered distribution given observations up to and
    including t, never anything after. That is exactly the no-lookahead property
    the project demands: a day's regime label can only depend on the past.

    Because the fitted transition matrix is very "sticky" (high diagonal),
    forward filtering also produces far more persistent regimes than decoding
    short windows in isolation, which keeps turnover realistic.
    """
    log_emission = model._compute_log_likelihood(X_scaled)         # (T, n_states)
    log_transition = np.log(np.clip(model.transmat_, 1e-300, None))
    log_start = np.log(np.clip(model.startprob_, 1e-300, None))

    n_obs, n_states = log_emission.shape
    log_alpha = np.empty((n_obs, n_states))
    log_alpha[0] = log_start + log_emission[0]
    for t in range(1, n_obs):
        # forward recursion: combine previous filtered beliefs with transitions
        log_alpha[t] = log_emission[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_transition, axis=0)
    return np.argmax(log_alpha, axis=1)


def regimes_from_states(states, label_map):
    """Translate raw state indices into Bull/Bear/Crisis names."""
    return np.array([label_map[int(s)] for s in states])


def predict_regimes(model, label_map, X_scaled):
    """Viterbi-decode regime names for each row of ``X_scaled``.

    Viterbi uses the whole sequence, so this is only appropriate for an
    in-sample visualization of the regimes. The leak-free walk-forward path uses
    ``filter_states`` instead.
    """
    return regimes_from_states(model.predict(X_scaled), label_map)
