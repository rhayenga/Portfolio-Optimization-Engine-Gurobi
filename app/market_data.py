"""Fetch prices and estimate mean returns / covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS = 252


def fetch_adjusted_closes(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    """Download daily adjusted closes for ``lookback_days`` of calendar history."""
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


def estimate_mu_sigma(closes: pd.DataFrame) -> tuple[list[str], list[float], list[list[float]]]:
    """Annualized mean returns and covariance from daily log returns."""
    tickers = [str(c) for c in closes.columns]
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if log_ret.empty:
        raise ValueError("could not compute returns from price history")

    mu = (log_ret.mean() * TRADING_DAYS).tolist()
    cov = (log_ret.cov() * TRADING_DAYS).values
    # Numerical harden: symmetrize
    cov = 0.5 * (cov + cov.T)
    return tickers, [float(x) for x in mu], [[float(x) for x in row] for row in cov]
