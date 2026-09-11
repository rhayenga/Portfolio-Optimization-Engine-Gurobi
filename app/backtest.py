"""Optimize on a train window and evaluate buy-and-hold weights out of sample."""

from __future__ import annotations

import math

import numpy as np

from app.market_data import TRADING_DAYS, estimate_mu_sigma, fetch_adjusted_closes
from app.models import BacktestRequest, BacktestResponse, OptimizeRequest
from app.optimizer import optimize_mean_variance


def _performance(daily_returns: np.ndarray) -> dict[str, float]:
    if daily_returns.size == 0:
        raise ValueError("no test-period returns")
    wealth = np.cumprod(1.0 + daily_returns)
    total_return = float(wealth[-1] - 1.0)
    n = len(daily_returns)
    ann_factor = TRADING_DAYS / max(n, 1)
    ann_return = float(wealth[-1] ** ann_factor - 1.0)
    ann_vol = float(daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else 0.0
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    max_dd = float(drawdown.min())
    return {
        "total_return": round(total_return, 8),
        "annualized_return": round(ann_return, 8),
        "annualized_volatility": round(ann_vol, 8),
        "sharpe_ratio": round(sharpe, 8),
        "max_drawdown": round(max_dd, 8),
    }


def run_backtest(req: BacktestRequest) -> BacktestResponse:
    total_days = req.train_days + req.test_days + 5
    closes = fetch_adjusted_closes(req.tickers, total_days)
    if len(closes) < req.train_days + req.test_days:
        raise ValueError(
            f"need at least {req.train_days + req.test_days} trading days; got {len(closes)}"
        )

    train = closes.iloc[-(req.train_days + req.test_days) : -req.test_days]
    test = closes.iloc[-req.test_days :]
    tickers, mu, cov, intensity = estimate_mu_sigma(train, shrink=req.shrink_covariance)

    opt_req = OptimizeRequest(
        tickers=tickers,
        expected_returns=mu,
        covariance=cov,
        risk_aversion=req.risk_aversion,
        max_weight=req.max_weight,
        min_weight=req.min_weight,
        long_only=req.long_only,
        min_return=req.min_return,
        max_assets=req.max_assets,
        sectors=req.sectors,
        sector_limits=req.sector_limits,
        current_weights=req.current_weights,
        max_turnover=req.max_turnover,
    )
    sol = optimize_mean_variance(opt_req)
    weights = np.asarray(sol.weights, dtype=float)

    simple = test.pct_change().dropna()
    port_daily = simple.values @ weights
    eq_daily = simple.values.mean(axis=1)

    optimized = _performance(port_daily)
    equal_weight = _performance(eq_daily)

    return BacktestResponse(
        tickers=tickers,
        train_days=req.train_days,
        test_days=len(simple),
        weights=sol.weights,
        holdings_count=sol.holdings_count,
        optimized=optimized,
        equal_weight=equal_weight,
        expected_returns=[round(x, 8) for x in mu],
        shrinkage_intensity=round(intensity, 8),
        status="optimal",
        solver="gurobi",
    )
