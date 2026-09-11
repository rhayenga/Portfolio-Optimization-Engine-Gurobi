"""Fetch prices and estimate mean returns / covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS = 252


def fetch_adjusted_closes(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    """Download daily adjusted closes for ``lookback_days`` of calendar history."""
    if not tickers:
        raise ValueError("tickers must be a non-empty list")
    if lookback_days < 30:
        raise ValueError("lookback_days must be at least 30")

    # Pad calendar days so we still get ~lookback trading observations
    period_days = int(lookback_days * 1.6) + 30
    data = yf.download(
        tickers=tickers,
        period=f"{period_days}d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if data.empty:
        raise ValueError("no price data returned for the requested tickers")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("downloaded data missing Close prices")
        closes = data["Close"].copy()
    else:
        # Single ticker: flat columns
        if "Close" not in data.columns:
            raise ValueError("downloaded data missing Close prices")
        closes = data[["Close"]].copy()
        closes.columns = [tickers[0]]

    closes = closes.dropna(how="all")
    missing = [t for t in tickers if t not in closes.columns]
    if missing:
        raise ValueError(f"missing price data for: {', '.join(missing)}")

    closes = closes[tickers].dropna(how="any")
    if len(closes) < 20:
        raise ValueError("not enough overlapping price history after cleaning")

    # Keep the most recent lookback trading days when available
    if len(closes) > lookback_days:
        closes = closes.iloc[-lookback_days:]
    return closes


def ledoit_wolf_shrinkage(sample_cov: np.ndarray, n_obs: int) -> tuple[np.ndarray, float]:
    """
    Ledoit–Wolf shrinkage of a sample covariance toward a scaled identity.

    Returns (shrunk_cov, shrinkage_intensity).
    """
    x = np.asarray(sample_cov, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("sample_cov must be square")
    p = x.shape[0]
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")

    # Target: mu * I with mu = average variance
    mu = float(np.trace(x) / p)
    prior = mu * np.eye(p)

    # Compact LW intensity using Frobenius norms (classic LW identity prior)
    d = x - prior
    delta = float(np.sum(d * d))
    # Bound intensity in [0, 1]; use simple unbiased-style formula
    # beta ≈ ||S||_F^2 / n , alpha = ||S - F||_F^2 ; kappa = beta/alpha
    beta = float(np.sum(x * x)) / n_obs
    alpha = delta
    if alpha < 1e-18:
        intensity = 0.0
    else:
        intensity = max(0.0, min(1.0, beta / alpha))

    shrunk = intensity * prior + (1.0 - intensity) * x
    shrunk = 0.5 * (shrunk + shrunk.T)
    return shrunk, intensity


def estimate_mu_sigma(
    closes: pd.DataFrame,
    *,
    shrink: bool = True,
) -> tuple[list[str], list[float], list[list[float]], float]:
    """
    Annualized mean returns and covariance from daily log returns.

    When ``shrink`` is True, applies Ledoit–Wolf shrinkage toward scaled identity.
    Returns (tickers, mu, cov, shrinkage_intensity).
    """
    tickers = [str(c) for c in closes.columns]
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if log_ret.empty:
        raise ValueError("could not compute returns from price history")

    mu = (log_ret.mean() * TRADING_DAYS).tolist()
    sample = (log_ret.cov() * TRADING_DAYS).values
    sample = 0.5 * (sample + sample.T)

    intensity = 0.0
    cov = sample
    if shrink:
        cov, intensity = ledoit_wolf_shrinkage(sample, n_obs=len(log_ret))

    return tickers, [float(x) for x in mu], [[float(x) for x in row] for row in cov], float(intensity)
