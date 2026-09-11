"""Command-line entrypoints for optimize / frontier / backtest."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys

from app.backtest import run_backtest
from app.market_data import estimate_mu_sigma, fetch_adjusted_closes
from app.models import (
    BacktestRequest,
    EfficientFrontierRequest,
    OptimizeRequest,
)
from app.optimizer import compute_efficient_frontier, optimize_mean_variance


def _parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _parse_sectors(raw: str | None, n: int) -> list[str] | None:
    if not raw:
        return None
    sectors = [s.strip().upper() for s in raw.split(",")]
    if len(sectors) != n:
        raise SystemExit("sectors count must match tickers count")
    return sectors


def _quiet(fn, *args, **kwargs):
    """Run callable while swallowing Gurobi license banners on stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def cmd_optimize(args: argparse.Namespace) -> None:
    tickers = _parse_tickers(args.tickers)
    closes = fetch_adjusted_closes(tickers, args.lookback_days)
    tickers, mu, cov, _intensity = estimate_mu_sigma(closes)
    req = OptimizeRequest(
        tickers=tickers,
        expected_returns=mu,
        covariance=cov,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        long_only=not args.allow_short,
        max_assets=args.max_assets,
        sectors=_parse_sectors(args.sectors, len(tickers)),
        sector_limits=json.loads(args.sector_limits) if args.sector_limits else None,
    )
    out = _quiet(optimize_mean_variance, req)
    print(json.dumps(out.model_dump(), indent=2))


def cmd_frontier(args: argparse.Namespace) -> None:
    tickers = _parse_tickers(args.tickers)
    closes = fetch_adjusted_closes(tickers, args.lookback_days)
    tickers, mu, cov, _intensity = estimate_mu_sigma(closes)
    req = EfficientFrontierRequest(
        tickers=tickers,
        expected_returns=mu,
        covariance=cov,
        n_points=args.n_points,
        max_weight=args.max_weight,
        long_only=not args.allow_short,
        max_assets=args.max_assets,
    )
    out = _quiet(compute_efficient_frontier, req)
    print(json.dumps(out.model_dump(), indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    tickers = _parse_tickers(args.tickers)
    req = BacktestRequest(
        tickers=tickers,
        train_days=args.train_days,
        test_days=args.test_days,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        long_only=not args.allow_short,
        max_assets=args.max_assets,
    )
    out = _quiet(run_backtest, req)
    print(json.dumps(out.model_dump(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="portfolio-opt", description="Gurobi portfolio optimization CLI")
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize", help="Estimate μ/Σ from market data and optimize")
    opt.add_argument("--tickers", required=True, help="Comma-separated tickers")
    opt.add_argument("--lookback-days", type=int, default=252)
    opt.add_argument("--risk-aversion", type=float, default=2.0)
    opt.add_argument("--max-weight", type=float, default=1.0)
    opt.add_argument("--max-assets", type=int, default=None)
    opt.add_argument("--sectors", default=None, help="Comma-separated sectors aligned to tickers")
    opt.add_argument("--sector-limits", default=None, help='JSON e.g. {"TECH":0.7}')
    opt.add_argument("--allow-short", action="store_true")
    opt.set_defaults(func=cmd_optimize)

    fr = sub.add_parser("frontier", help="Efficient frontier from market data")
    fr.add_argument("--tickers", required=True)
    fr.add_argument("--lookback-days", type=int, default=252)
    fr.add_argument("--n-points", type=int, default=10)
    fr.add_argument("--max-weight", type=float, default=1.0)
    fr.add_argument("--max-assets", type=int, default=None)
    fr.add_argument("--allow-short", action="store_true")
    fr.set_defaults(func=cmd_frontier)

    bt = sub.add_parser("backtest", help="Train/optimize then evaluate out of sample")
    bt.add_argument("--tickers", required=True)
    bt.add_argument("--train-days", type=int, default=252)
    bt.add_argument("--test-days", type=int, default=63)
    bt.add_argument("--risk-aversion", type=float, default=2.0)
    bt.add_argument("--max-weight", type=float, default=1.0)
    bt.add_argument("--max-assets", type=int, default=None)
    bt.add_argument("--allow-short", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
