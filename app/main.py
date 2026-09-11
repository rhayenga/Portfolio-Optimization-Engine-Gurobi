"""FastAPI entrypoint for the portfolio optimization engine."""

from fastapi import FastAPI, HTTPException

from app import __version__
from app.backtest import run_backtest
from app.market_data import estimate_mu_sigma, fetch_adjusted_closes
from app.models import (
    BacktestRequest,
    BacktestResponse,
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    MarketEfficientFrontierRequest,
    MarketEfficientFrontierResponse,
    MarketOptimizeRequest,
    MarketOptimizeResponse,
    OptimizeRequest,
    OptimizeResponse,
)
from app.optimizer import compute_efficient_frontier, optimize_mean_variance

app = FastAPI(
    title="Portfolio Optimization Engine",
    description=(
        "Mean-variance portfolio optimization powered by Gurobi. "
        "Optimize, sweep an efficient frontier, or backtest optimized weights out of sample."
    ),
    version=__version__,
)


def _run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    try:
        return optimize_mean_variance(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # gurobipy license / solver errors
        raise HTTPException(status_code=500, detail=f"solver error: {exc}") from exc


def _run_frontier(req: EfficientFrontierRequest) -> EfficientFrontierResponse:
    try:
        return compute_efficient_frontier(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"solver error: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "portfolio-optimization-engine", "version": __version__}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    return _run_optimize(req)


@app.post("/optimize/from-market", response_model=MarketOptimizeResponse)
def optimize_from_market(req: MarketOptimizeRequest) -> MarketOptimizeResponse:
    try:
        closes = fetch_adjusted_closes(req.tickers, req.lookback_days)
        tickers, mu, cov, intensity = estimate_mu_sigma(closes, shrink=req.shrink_covariance)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"market data error: {exc}") from exc

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
    base = _run_optimize(opt_req)
    return MarketOptimizeResponse(
        **base.model_dump(),
        lookback_days=req.lookback_days,
        expected_returns=[round(x, 8) for x in mu],
        covariance=[[round(x, 10) for x in row] for row in cov],
        shrinkage_intensity=round(intensity, 8),
    )


@app.post("/frontier", response_model=EfficientFrontierResponse)
def frontier(req: EfficientFrontierRequest) -> EfficientFrontierResponse:
    return _run_frontier(req)


@app.post("/frontier/from-market", response_model=MarketEfficientFrontierResponse)
def frontier_from_market(req: MarketEfficientFrontierRequest) -> MarketEfficientFrontierResponse:
    try:
        closes = fetch_adjusted_closes(req.tickers, req.lookback_days)
        tickers, mu, cov, intensity = estimate_mu_sigma(closes, shrink=req.shrink_covariance)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"market data error: {exc}") from exc

    frontier_req = EfficientFrontierRequest(
        tickers=tickers,
        expected_returns=mu,
        covariance=cov,
        n_points=req.n_points,
        max_weight=req.max_weight,
        min_weight=req.min_weight,
        long_only=req.long_only,
        max_assets=req.max_assets,
        sectors=req.sectors,
        sector_limits=req.sector_limits,
    )
    base = _run_frontier(frontier_req)
    return MarketEfficientFrontierResponse(
        **base.model_dump(),
        lookback_days=req.lookback_days,
        expected_returns=[round(x, 8) for x in mu],
        covariance=[[round(x, 10) for x in row] for row in cov],
        shrinkage_intensity=round(intensity, 8),
    )


@app.post("/backtest/from-market", response_model=BacktestResponse)
def backtest_from_market(req: BacktestRequest) -> BacktestResponse:
    try:
        return run_backtest(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"backtest error: {exc}") from exc
